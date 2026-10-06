"""Candidate generation (blocking) with polars joins, per country and per source.

Key kinds (a pair of records is a candidate if they share any key):
  nw  a number in the address + one of the record's 4 rarest address words
  np  the record's 2 rarest core-name tokens, as a pair
  nt  a single core-name token that is very rare (doc. freq <= 30)
  cc  the whole core name without spaces (website-style names)
  ap  the record's 2 rarest address words, as a pair
  nn  an address number (first 3) + one of the record's 2 rarest name tokens
  hw  the full structured house number ("2-1" for "B-2/1") + a rare address word
  nr  a core-name token with doc. freq <= 300; on the Source-2/3 side only for
      records whose address has no number (their only way to be found)
  ca  (ER_CA=1) the whole core name without spaces + one of the record's 2 rarest
      address words: common names found within one locality (India lost 41k true
      pairs whose identical name is shared by 6+ businesses)
With ER_REV=r, each record also keeps its r best S1 candidates (reverse retrieval), so a
record that ranks below the K kept by its true S1 survives when that S1 is its own best.
"Rarest" only considers tokens seen in at least 2 records: a token seen once is
usually a typo and can never be shared, so picking it would waste the key.
Every key block larger than its cap is skipped. The pre-score of a pair is the sum
of 1/log2(1 + block size) over its shared keys; the K best per S1 and source are
kept, and `kinds` records which key kinds the pair shared (a bitmask).
Token frequencies come from the split's own text (unsupervised).
"""
import time

import polars as pl

import config
import rules

K_PER_SOURCE = config.K_PER_SOURCE
CAPS = {k: int(v * config.CAP_MULT) for k, v in
        {"nw": 200, "np": 200, "nt": 30, "cc": 50, "ap": 200, "nn": 200, "hw": 200, "nr": 300, "ca": 200}.items()}
KIND_BITS = {"nw": 1, "np": 2, "nt": 4, "cc": 8, "ap": 16, "nn": 32, "hw": 64, "nr": 128, "em": 256, "ca": 512}
S1_CAP = int(200 * config.CAP_MULT)
_GENERIC = sorted(rules.ADDR_GENERIC)


def _tokens(col):
    return (pl.col(col).str.split(" ")
            .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique())


def _doc_freq(frames, col):
    ex = pl.concat([f.select(t=_tokens(col)) for f in frames]).explode("t").drop_nulls()
    return ex.group_by("t").len().rename({"len": "df"})


def _rarest(f, col, dfreq, k, exclude=None, min_df=2):
    ex = f.select("i", t=_tokens(col)).explode("t").drop_nulls()
    if exclude:
        ex = ex.filter(~pl.col("t").is_in(exclude))
    ex = ex.join(dfreq, on="t").filter(pl.col("df") >= min_df)
    return ex.sort(["i", "df", "t"]).group_by("i", maintain_order=True).head(k)


def _pair_key(rows, prefix):
    return (rows.sort(["i", "t"]).group_by("i")
            .agg(pl.col("t").str.join("|"), c=pl.len())
            .filter(pl.col("c") == 2).select("i", key=pl.lit(prefix) + pl.col("t")))


def keys(f, dfw, dfn, s1_side):
    """(i, key) rows for a normalized frame carrying a row index `i`."""
    words = _rarest(f, "ad", dfw, 4, _GENERIC)
    nums = (f.select("i", n=pl.col("nums").str.split(" ")
                     .list.eval(pl.element().filter(pl.element() != ""))
                     .list.unique(maintain_order=True).list.head(3))
            .explode("n").drop_nulls())
    names = _rarest(f, "core", dfn, 2)
    rare3 = _rarest(f, "core", dfn, 3).filter(pl.col("df") <= CAPS["nr"])
    if not s1_side:
        rare3 = rare3.join(f.filter(pl.col("nums") == "").select("i"), on="i")
    hn = f.filter(pl.col("hn").str.contains("-")).select("i", "hn")
    parts = [
        nums.join(words.select("i", "t"), on="i")
            .select("i", key=pl.lit("nw:") + pl.col("n") + "|" + pl.col("t")),
        _pair_key(names, "np:"),
        names.filter(pl.col("df") <= CAPS["nt"]).select("i", key=pl.lit("nt:") + pl.col("t")),
        f.filter(pl.col("cc").str.len_chars() >= 6).select("i", key=pl.lit("cc:") + pl.col("cc")),
        _pair_key(words.group_by("i", maintain_order=True).head(2), "ap:"),
        nums.join(names.select("i", "t"), on="i")
            .select("i", key=pl.lit("nn:") + pl.col("n") + "|" + pl.col("t")),
        hn.join(words.select("i", "t"), on="i")
            .select("i", key=pl.lit("hw:") + pl.col("hn") + "|" + pl.col("t")),
        rare3.select("i", key=pl.lit("nr:") + pl.col("t")),
    ]
    if config.USE_CA:
        parts.append(f.filter(pl.col("cc").str.len_chars() >= 4).select("i", "cc")
                     .join(words.group_by("i", maintain_order=True).head(2).select("i", "t"), on="i")
                     .select("i", key=pl.lit("ca:") + pl.col("cc") + "|" + pl.col("t")))
    return pl.concat(parts).unique()


def _cap_expr():
    kind = pl.col("key").str.slice(0, 2)
    expr = pl.lit(0)
    for k, cap in CAPS.items():
        expr = pl.when(kind == k).then(cap).otherwise(expr)
    return expr


def _truth_pairs(split, s1):
    """(i1, src, i2) of the ground truth, or None when the split has no labels."""
    path = config.DATA_DIR / split / f"{split}_ground_truth.tsv"
    if not path.exists():
        return None
    from data_io import read_ground_truth
    truth = read_ground_truth(path)
    gt = pl.DataFrame({"s1": [s for s, v in truth.items() for _ in v],
                       "id": [x for v in truth.values() for x in v]}, schema={"s1": pl.String, "id": pl.String})
    ids1 = pl.read_parquet(config.norm_path(split, 1)).select(s1="id").with_row_index("i1")
    ids2 = pl.concat([pl.read_parquet(config.norm_path(split, k)).select("id").with_row_index("i2")
                      .with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    return gt.join(ids1, on="s1").join(ids2, on="id").select("i1", "src", "i2")


# per-kind evidence columns fed to the rankers; "ca" only when enabled, since the saved
# stage-0/1 rankers were fitted without it
_W_KINDS = {k: b for k, b in KIND_BITS.items() if k != "em" and (k != "ca" or config.CA_RANK)}
W_COLS = [f"w_{k}" for k in _W_KINDS]
STAGE0_FEATS = ["pre", "nk", "n_union", "r_pre", "rk_pre", *W_COLS]


K0 = config.K0  # stage-0 shortlist per S1 and source, re-ranked by the string-aware stage-1 ranker
STAGE1_FEATS = [*STAGE0_FEATS, "s0", "c_tset", "a_tset", "hmain_eq", "n_inter"]


def stage0_path():
    return config.WORK_DIR / "model_stage0.txt"


def stage1_path():
    return config.WORK_DIR / "model_stage1.txt"


def _rec(f, side):
    """Strings a pair needs for the cheap similarities, keyed by row index."""
    return f.select(pl.col("i").alias(f"i{side}"), pl.col("core").alias(f"core_{side}"), pl.col("adf").alias(f"adf_{side}"),
                    pl.col("hmain").alias(f"hmain_{side}"),
                    pl.col("nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias(f"nt_{side}"))


def _cheap(pairs, r1, r2):
    """Four cheap similarities (name/address token-set, same main number, shared numbers)."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    j = pairs.select("i1", "i2").join(r1, on="i1", how="left").join(r2, on="i2", how="left")
    c = cpdist(j["core_1"].to_list(), j["core_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype="float32")
    a = cpdist(j["adf_1"].to_list(), j["adf_2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype="float32")
    return pairs.with_columns(
        c_tset=pl.Series(c), a_tset=pl.Series(a),
        hmain_eq=((j["hmain_1"] == j["hmain_2"]) & (j["hmain_1"] != "")).cast(pl.Int8),
        n_inter=j["nt_1"].list.set_intersection(j["nt_2"]).list.len().cast(pl.Float32))


def _shortlist(pairs, r0, r1_rank, rec1, rec2):
    """Stage 0 (block evidence) keeps K0 per S1; stage 1 (+ cheap strings) scores them."""
    pairs = pairs.with_columns(s0=pl.Series(r0.predict(pairs.select(STAGE0_FEATS).to_numpy()), dtype=pl.Float32))
    pairs = pairs.filter(pl.col("s0").rank("ordinal", descending=True).over("i1") <= K0)
    if r1_rank is None:
        return pairs, pairs["s0"]
    pairs = _cheap(pairs, rec1, rec2)
    return pairs, pl.Series(r1_rank.predict(pairs.select(STAGE1_FEATS).to_numpy()), dtype=pl.Float32)


def _union(k1_chunk, k2):
    """Every S1-candidate pair sharing a key, with its evidence per key kind."""
    w = 1.0 / (pl.col("len") + 1).log(2)
    return (k1_chunk.join(k2, on="key").group_by("i1", "i2")
            .agg(pre=w.sum().cast(pl.Float32), nk=pl.len().cast(pl.UInt16), kinds=pl.col("b").bitwise_or(),
                 **{f"w_{k}": pl.when(pl.col("b") == b).then(w).otherwise(0.0).sum().cast(pl.Float32)
                    for k, b in _W_KINDS.items()})
            .with_columns(n_union=pl.len().over("i1").cast(pl.Float32),
                          r_pre=(pl.col("pre") / pl.col("pre").max().over("i1")).cast(pl.Float32),
                          rk_pre=pl.col("pre").rank("ordinal", descending=True).over("i1").cast(pl.Float32)))


def block_split(split, k_per_source=K_PER_SOURCE, chunk=200_000, sample=None, write=True, rev=None,
                diag=False, countries=None):
    """Candidates of a split. The K kept per S1 and source are chosen by the pre-score,
    or -- when the learned rankers exist in work/ -- by stage 0 (block evidence, keeps
    K0) followed by stage 1 (block evidence + cheap string similarities). With rev=r
    (default config.REV), each record's r best S1 candidates are kept as well.
    With `sample` (S1 row ids), returns labelled pairs of those S1 for ranker training:
    the full union if stage 0 is not trained yet, else the stage-0 shortlist with the
    cheap similarities (training data for stage 1).
    diag=True (with write=False) reports recall at every step -- union (with / without the
    ca key), stage-0 shortlist, the K kept, and K kept + reverse top-r for r = 1..3 -- for
    the `countries` given (default all)."""
    rev = config.REV if rev is None else rev
    rev_max = max(rev, 3 if diag else 0)
    t0 = time.time()
    cols = ["country", "core", "cc", "ad", "adf", "nums", "hn", "hmain"]
    s1 = pl.read_parquet(config.norm_path(split, 1)).select(cols).with_row_index("i")
    oth = {k: pl.read_parquet(config.norm_path(split, k)).select(cols).with_row_index("i") for k in (2, 3)}
    truth = _truth_pairs(split, s1)
    ranker = rank1 = None
    if stage0_path().exists():
        import lightgbm as lgb
        ranker = lgb.Booster(model_file=str(stage0_path()))
        if sample is None and stage1_path().exists():
            rank1 = lgb.Booster(model_file=str(stage1_path()))
    hits, rev_hits = [], []
    out = []
    for country in sorted(s1["country"].unique().to_list()):
        if countries and country not in countries:
            continue
        f1 = s1.filter(pl.col("country") == country)
        fo = {k: v.filter(pl.col("country") == country) for k, v in oth.items()}
        dfw, dfn = _doc_freq([f1, *fo.values()], "ad"), _doc_freq([f1, *fo.values()], "core")
        k1 = keys(f1, dfw, dfn, s1_side=True).rename({"i": "i1"})
        k1 = k1.join(k1.group_by("key").len().filter(pl.col("len") <= S1_CAP).select("key"), on="key")
        if sample is not None:
            k1 = k1.filter(pl.col("i1").is_in(sample.implode()))
        rec1 = _rec(f1, 1)
        for src, f in fo.items():
            k2 = keys(f, dfw, dfn, s1_side=False).rename({"i": "i2"})
            blk = k2.group_by("key").len().filter(pl.col("len") <= _cap_expr())
            k2 = k2.join(blk, on="key").with_columns(
                b=pl.col("key").str.slice(0, 2).replace_strict(KIND_BITS, return_dtype=pl.UInt16))
            rec2 = _rec(f, 2)
            ids = f1["i"]
            cols = ["i1", "i2", "pre", "nk", "kinds", *(W_COLS if ranker is not None else [])]
            near_all, near_full = [], []
            for start in range(0, f1.height, chunk):
                lo, hi = ids[start], ids[min(start + chunk, f1.height) - 1]
                pairs = _union(k1.filter(pl.col("i1").is_between(lo, hi)), k2)
                full = pairs
                if ranker is not None and pairs.height:
                    pairs, score = _shortlist(pairs, ranker, rank1, rec1, rec2)
                    if sample is not None and rank1 is None:  # stage-1 training data
                        pairs = _cheap(pairs, rec1, rec2)
                else:
                    score = pairs["pre"]
                if sample is not None:
                    lab = truth.filter(pl.col("src") == src).select("i1", "i2", y=pl.lit(1, pl.Int8))
                    out.append(pairs.join(lab, on=["i1", "i2"], how="left")
                               .with_columns(pl.col("y").fill_null(0), src=pl.lit(src, pl.Int8)))
                    continue
                scored = pairs.with_columns(sc=score)
                scored = scored.with_columns(
                    fwd=pl.col("sc").rank("ordinal", descending=True).over("i1") <= k_per_source)
                top = scored.filter(pl.col("fwd")).select(cols)
                out.append(top.with_columns(src=pl.lit(src, pl.Int8)))
                if rev_max:
                    # a record's global top-r among all S1 is inside some chunk's top-r, so keeping
                    # each chunk's top-rev_max per record is enough for the global selection below
                    near = scored.filter(pl.col("sc").rank("ordinal", descending=True).over("i2") <= rev_max)
                    near_all.append(near.select("i1", "i2", "sc", "fwd"))
                    near_full.append(near.filter(~pl.col("fwd")).select(cols))
                if truth is not None:
                    tt = truth.filter((pl.col("src") == src) & pl.col("i1").is_between(lo, hi)
                                      & pl.col("i1").is_in(ids.implode()))
                    found = lambda fr: fr.join(tt, on=["i1", "i2"]).height
                    hits.append((country, src, tt.height, found(full),
                                 found(full.filter((pl.col("kinds") & 511) > 0)), found(pairs), found(top)))
            if rev_max and near_all:
                g = (pl.concat(near_all)
                     .with_columns(rr=pl.col("sc").rank("ordinal", descending=True).over("i2")))
                if rev:
                    add = g.filter((pl.col("rr") <= rev) & ~pl.col("fwd")).select("i1", "i2")
                    extra = add.join(pl.concat(near_full).unique(["i1", "i2"]), on=["i1", "i2"])
                    out.append(extra.with_columns(src=pl.lit(src, pl.Int8)))
                if diag and truth is not None:
                    tc = truth.filter((pl.col("src") == src) & pl.col("i1").is_in(ids.implode()))
                    for r in (1, 2, 3):
                        a = g.filter((pl.col("rr") <= r) & ~pl.col("fwd"))
                        rev_hits.append((country, src, r, a.height, a.join(tc, on=["i1", "i2"]).height))
            print(f"  {split} {country} S{src}: done ({time.time()-t0:.0f}s)", flush=True)
    if sample is not None:
        return pl.concat(out)
    if hits:
        d = (pl.DataFrame(hits, schema=["country", "src", "true", "union", "union_no_ca", "shortlist", "top"],
                          orient="row")
             .group_by("country", "src").sum().sort("country", "src"))
        for c in ("union", "union_no_ca", "shortlist", "top"):
            d = d.with_columns((pl.col(c) / pl.col("true")).alias(f"{c}_recall"))
        print(d)
        tot = d["true"].sum()
        print(f"overall: union recall {d['union'].sum() / tot:.4f} (without ca {d['union_no_ca'].sum() / tot:.4f}) | "
              f"shortlist {d['shortlist'].sum() / tot:.4f} | kept {d['top'].sum() / tot:.4f} | kept by "
              f"{'stage-0+1 rankers' if rank1 is not None else 'stage-0 ranker' if ranker is not None else 'pre-score'}")
        if rev_hits:
            n_s1 = s1.filter(pl.col("country").is_in(countries)).height if countries else s1.height
            rv = pl.DataFrame(rev_hits, schema=["country", "src", "r", "added", "added_true"], orient="row")
            for r in (1, 2, 3):
                x = rv.filter(pl.col("r") == r)
                print(f"  + reverse top-{r}: recall {(d['top'].sum() + x['added_true'].sum()) / tot:.4f} "
                      f"(+{x['added_true'].sum() / tot:.4f}) for +{x['added'].sum() / n_s1:.2f} candidates per S1")
    cands = pl.concat(out).select("i1", "src", "i2", "pre", "nk", "kinds", *(W_COLS if ranker is not None else []))
    if write:
        path = config.WORK_DIR / "cands" / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        cands.write_parquet(path)
    print(f"{split}: {cands.height} candidate pairs, {cands.height / s1.height:.1f} per S1 ({time.time()-t0:.0f}s)")
    return cands


def train_stage0(n_s1=200_000, k=K_PER_SOURCE):
    """Learn which union pairs to keep: LightGBM on per-key-kind evidence, fitted on the
    full union of a sample of training S1 (half to fit, half to measure the recall gain)."""
    import lightgbm as lgb
    import numpy as np
    t0 = time.time()
    n = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).height
    ids = np.random.default_rng(config.SEED + 3).choice(n, n_s1, replace=False).astype(np.uint32)
    u = block_split("train", sample=pl.Series(ids))
    fit_ids = pl.Series(ids[: n_s1 // 2])
    fit = u.filter(pl.col("i1").is_in(fit_ids.implode()))
    ev = u.filter(~pl.col("i1").is_in(fit_ids.implode()))
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, verbose=-1,
                  num_threads=config.N_WORKERS + 2, seed=config.SEED)
    booster = lgb.train(params, lgb.Dataset(fit.select(STAGE0_FEATS).to_numpy(), label=fit["y"].to_numpy(),
                                            feature_name=STAGE0_FEATS), 300)
    ev = ev.with_columns(s0=pl.Series(booster.predict(ev.select(STAGE0_FEATS).to_numpy()), dtype=pl.Float32))
    ev_true = _truth_pairs("train", None).filter(pl.col("i1").is_in(pl.Series(ids[n_s1 // 2:]).implode())).height
    for name in ("pre", "s0"):
        kept = ev.filter(pl.col(name).rank("ordinal", descending=True).over("i1", "src") <= k)
        print(f"  top-{k} per source by {name}: recall {kept['y'].sum() / ev_true:.4f}")
    print(f"  union recall {ev['y'].sum() / ev_true:.4f} | union size {ev.height / (n_s1 // 2):.1f} per S1")
    booster.save_model(str(stage0_path()))
    print(f"stage-0 ranker saved ({time.time()-t0:.0f}s)")


def load_cands(split):
    """Blocking candidates, plus the embedding-retrieved pairs (kind `em`) when enabled."""
    cands = pl.read_parquet(config.WORK_DIR / "cands" / f"{split}.parquet")
    em_file = config.WORK_DIR / "cands" / f"{split}_em.parquet"
    if not (config.USE_EMB and em_file.exists()):
        return cands
    em = pl.read_parquet(em_file).select("i1", "src", "i2", e=pl.lit(True))
    return (cands.join(em, on=["i1", "src", "i2"], how="full", coalesce=True)
            .with_columns(pre=pl.col("pre").fill_null(0.0), nk=pl.col("nk").fill_null(0),
                          kinds=(pl.col("kinds").fill_null(0)
                                 | pl.when(pl.col("e")).then(KIND_BITS["em"]).otherwise(0)).cast(pl.UInt16))
            .drop("e"))


def recall_report(split="train"):
    """Pair recall / entity coverage of the candidate set against the ground truth."""
    from data_io import read_ground_truth
    cands = pl.read_parquet(config.WORK_DIR / "cands" / f"{split}.parquet")
    s1 = pl.read_parquet(config.norm_path(split, 1)).select("id", "country").with_row_index("i1")
    idx = {k: pl.read_parquet(config.norm_path(split, k)).select("id").with_row_index("i2")
           .with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)}
    truth = read_ground_truth(config.DATA_DIR / split / f"{split}_ground_truth.tsv")
    gt = pl.DataFrame({"s1": [s for s, v in truth.items() for _ in v],
                       "id": [x for v in truth.values() for x in v]})
    gt = (gt.join(s1.rename({"id": "s1"}), on="s1")
          .join(pl.concat(idx.values()), on="id"))
    hit = gt.join(cands.select("i1", "src", "i2", found=pl.lit(True)), on=["i1", "src", "i2"], how="left")
    hit = hit.with_columns(pl.col("found").fill_null(False))
    print(hit.group_by("country", "src").agg(pair_recall=pl.col("found").mean(), pairs=pl.len()).sort("country", "src"))
    status = pl.concat([pl.read_parquet(config.norm_path(split, k), columns=["ad", "nums"]).with_row_index("i2")
                        .with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    status = status.select("src", "i2", addr=pl.when((pl.col("ad") == "") & (pl.col("nums") == "")).then(pl.lit("empty"))
                           .when(pl.col("nums") == "").then(pl.lit("no number")).otherwise(pl.lit("has number")))
    print(hit.join(status, on=["src", "i2"]).group_by("country", "addr")
          .agg(pair_recall=pl.col("found").mean(), pairs=pl.len()).sort("country", "addr"))
    ent = hit.group_by("i1").agg(all_found=pl.col("found").all())
    print(f"pair recall {hit['found'].mean():.4f} | entities with all matches found {ent['all_found'].mean():.4f} "
          f"| candidates per S1 {cands.height / s1.height:.1f}")


def train_stage1(n_s1=200_000, k=K_PER_SOURCE):
    """Stage 1 re-ranks the stage-0 shortlist (K0 per source) with cheap string
    similarities; fitted on the same sampled training S1 as stage 0."""
    import lightgbm as lgb
    import numpy as np
    t0 = time.time()
    n = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).height
    ids = np.random.default_rng(config.SEED + 3).choice(n, n_s1, replace=False).astype(np.uint32)
    u = block_split("train", sample=pl.Series(ids))
    fit_ids = pl.Series(ids[: n_s1 // 2])
    fit = u.filter(pl.col("i1").is_in(fit_ids.implode()))
    ev = u.filter(~pl.col("i1").is_in(fit_ids.implode()))
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, verbose=-1,
                  num_threads=config.N_WORKERS + 2, seed=config.SEED)
    booster = lgb.train(params, lgb.Dataset(fit.select(STAGE1_FEATS).to_numpy(), label=fit["y"].to_numpy(),
                                            feature_name=STAGE1_FEATS), 300)
    ev = ev.with_columns(s1=pl.Series(booster.predict(ev.select(STAGE1_FEATS).to_numpy()), dtype=pl.Float32))
    ev_true = _truth_pairs("train", None).filter(pl.col("i1").is_in(pl.Series(ids[n_s1 // 2:]).implode())).height
    for name in ("pre", "s0", "s1"):
        kept = ev.filter(pl.col(name).rank("ordinal", descending=True).over("i1", "src") <= k)
        print(f"  top-{k} per source by {name} (within the stage-0 top-{K0}): recall {kept['y'].sum() / ev_true:.4f}")
    print(f"  stage-0 shortlist recall {ev['y'].sum() / ev_true:.4f} | shortlist size {ev.height / (n_s1 // 2):.1f} per S1")
    booster.save_model(str(stage1_path()))
    print(f"stage-1 ranker saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["stage0"]:
        train_stage0()
        sys.exit()
    if sys.argv[1:2] == ["stage1"]:
        train_stage1()
        sys.exit()
    for sp in (sys.argv[1:] or ["train"]):
        block_split(sp)
        if sp == "train":
            recall_report("train")
