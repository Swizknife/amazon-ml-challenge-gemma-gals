"""Second-stage collective model: re-scores every pruned candidate pair with the
first-stage matcher's own predictions about competing and co-matched records.

This is stacked collective classification: the probabilities of neighbouring pairs
become features of a second model. Top Foursquare-matching solutions use the same
two-level boosting design, and TransClean-style transitivity checks follow the same idea.
Features per pair (i1, src, i2), all computed from the first-stage probability p:
  * entity view: rank of p within the entity (overall and per source), the entity's best
    other p, gap to the best, count of confident candidates, sum of p, list size;
  * record view: the best p any *other* entity gives this record, the gap to it,
    the number of competing entities;
  * transitivity: name / address token-set similarity and same main house number
    between this candidate and the entity's two most probable other candidates,
    plus their probabilities (a true match should agree with the entity's other matches).
The first-stage p is out-of-fold on train (train_oof.py) and comes from the final model
on test, exactly as the thresholds are tuned today. The stacker is fitted on two folds
of Source-1 entities, thresholds are re-tuned on the pruned out-of-fold predictions
(metablock.py grid), and a submission is written only if the weighted F0.5
(US 0.45 / India 0.55, the test mix) improves by at least MIN_GAIN.

    python stack.py [--sample-s1 N] [--dry-run]
Output (when it helps): <OUT_DIR>_stack/matching_results.tsv + candidate_pairs.tsv + stack_report.json

Final submission (v6): python train_pruned.py, then
    python stack.py --second oof_pruned --primary second --wide-grid --strict-unseen --iterate
(second first-stage opinion from train_pruned.py, two collective rounds; OOF US 0.9832 / India 0.9755).
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import blocking
import config
import metablock
from data_io import write_id_lists
from decode import decode
from metric import f05

KEYS = ["i1", "src", "i2"]
MIN_GAIN = 0.001
WEIGHTS = {"US": 0.45, "India": 0.55}
FEATS = ["p", "src_f", "p_rank", "p_max", "p_gap", "p_other", "n_cand", "n_hi", "p_sum", "p_rank_s", "n_hi_s",
         "r_n", "p_rival", "p_rival_gap", "a_c", "a_a", "a_h", "a_p", "b_c", "b_a", "b_h", "b_p"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=config.N_WORKERS + 2,
              verbose=-1, seed=config.SEED)
ROUNDS = 400
# France (test only, no labels): keep v5's stricter tuned setting rather than the looser v6 US/India values
UNSEEN = dict(t_single=0.65, t_match=0.3, miss_mass=0.34)
T0 = time.time()


def log(msg):
    print(f"[stack {time.time() - T0:6.0f}s] {msg}", flush=True)


def other_records(split):
    return pl.concat([pl.read_parquet(config.norm_path(split, k), columns=["id", "core", "adf", "hmain"])
                      .with_row_index("i2").with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])


def kept_train(sample):
    """Meta-blocking applied to the cached training pairs (as metablock._kept_train, optionally sampled)."""
    saved = config.artifact("kept_train", "parquet")  # written by train_pruned.py
    if saved.exists():
        kept = pl.read_parquet(saved)
        return kept if sample is None else kept.filter(pl.col("i1").is_in(sample.implode()))
    ranker = lgb.Booster(model_file=str(metablock.model_path()))
    cols = ranker.feature_name()
    scan = pl.scan_parquet(config.WORK_DIR / "feats_train" / "*.parquet").select(*KEYS, *cols)
    if sample is not None:
        scan = scan.filter(pl.col("i1").is_in(sample.implode()))
    X = scan.collect()
    score = np.concatenate([ranker.predict(X.slice(s, 5_000_000).select(cols).cast(pl.Float32).to_numpy())
                            for s in range(0, X.height, 5_000_000)])
    return metablock._keep(X.select(KEYS), score, metablock.TAU)


def blend(pairs, second, primary="mean"):
    """Two first-stage opinions, both kept as features; p becomes their mean (or the second one)."""
    pairs = pairs.join(second.select(*KEYS, p_second=pl.col("p").cast(pl.Float32)), on=KEYS, how="left")
    pairs = pairs.with_columns(p_first=pl.col("p"), p_second=pl.col("p_second").fill_null(pl.col("p")))
    p = pl.col("p_second") if primary == "second" else (pl.col("p_first") + pl.col("p_second")) / 2
    return pairs.with_columns(p=p.cast(pl.Float32))


def collective(pairs):
    """Entity-view and record-view features of the first-stage probability."""
    L = pl.col
    pairs = pairs.with_columns(
        p_rank=L("p").rank("ordinal", descending=True).over("i1").cast(pl.Float32),
        p_max=L("p").max().over("i1"),
        p_top2=L("p").top_k(2).min().over("i1"),
        n_cand=pl.len().over("i1").cast(pl.Float32),
        n_hi=(L("p") >= 0.5).sum().over("i1").cast(pl.Float32),
        p_sum=L("p").sum().over("i1"),
        p_rank_s=L("p").rank("ordinal", descending=True).over("i1", "src").cast(pl.Float32),
        n_hi_s=(L("p") >= 0.5).sum().over("i1", "src").cast(pl.Float32),
        r_n=pl.len().over("src", "i2").cast(pl.Float32),
        r_max=L("p").max().over("src", "i2"),
        r_top2=L("p").top_k(2).min().over("src", "i2"))
    pairs = pairs.with_columns(
        p_gap=L("p") - L("p_max"),
        p_other=pl.when(L("n_cand") == 1).then(-1.0).when(L("p_rank") == 1).then(L("p_top2")).otherwise(L("p_max")),
        p_rival=pl.when(L("r_n") == 1).then(-1.0).when(L("p") >= L("r_max")).then(L("r_top2")).otherwise(L("r_max")))
    return pairs.with_columns(p_rival_gap=L("p") - L("p_rival")).drop("p_top2", "r_max", "r_top2")


def transitivity(pairs, so, chunk=3_000_000):
    """Agreement between each candidate and the entity's two most probable other candidates."""
    L = pl.col
    top = pairs.filter(L("p_rank") <= 3).select("i1", "p_rank", "src", "i2", "p")
    for k in (1, 2, 3):
        pairs = pairs.join(top.filter(L("p_rank") == k).select("i1", **{f"t{k}s": "src", f"t{k}i": "i2", f"t{k}p": "p"}),
                           on="i1", how="left")
    r = L("p_rank")
    refs = pairs.select(*KEYS,
                        a_s=pl.when(r == 1).then("t2s").otherwise("t1s"), a_i=pl.when(r == 1).then("t2i").otherwise("t1i"),
                        a_p=pl.when(r == 1).then("t2p").otherwise("t1p"),
                        b_s=pl.when(r <= 2).then("t3s").otherwise("t2s"), b_i=pl.when(r <= 2).then("t3i").otherwise("t2i"),
                        b_p=pl.when(r <= 2).then("t3p").otherwise("t2p"))
    txt = so.select("src", "i2", "core", "adf", "hmain")
    parts = []
    for s in range(0, refs.height, chunk):
        ch = refs.slice(s, chunk).join(txt, on=["src", "i2"], how="left")
        cols = {}
        for tag in ("a", "b"):
            ch = ch.join(txt.rename({"src": f"{tag}_s", "i2": f"{tag}_i", "core": f"core_{tag}", "adf": f"adf_{tag}",
                                     "hmain": f"hm_{tag}"}), on=[f"{tag}_s", f"{tag}_i"], how="left")
            miss = ch[f"{tag}_i"].is_null().to_numpy()
            c = cpdist(ch["core"].fill_null("").to_list(), ch[f"core_{tag}"].fill_null("").to_list(),
                       scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            a = cpdist(ch["adf"].fill_null("").to_list(), ch[f"adf_{tag}"].fill_null("").to_list(),
                       scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            h = ((ch["hmain"] == ch[f"hm_{tag}"]) & (ch["hmain"] != "")).fill_null(False).cast(pl.Float32).to_numpy()
            c, a, h = np.array(c), np.array(a), np.array(h)  # writable copies
            c[miss], a[miss], h[miss] = np.nan, np.nan, np.nan
            cols |= {f"{tag}_c": c, f"{tag}_a": a, f"{tag}_h": h}
        parts.append(ch.select(*KEYS, "a_p", "b_p").with_columns(**{k: pl.Series(v) for k, v in cols.items()}))
    return pairs.drop([c for c in pairs.columns if c[:1] == "t" and c[1:2].isdigit()]).join(pl.concat(parts), on=KEYS)


def features(pairs, so):
    X = transitivity(collective(pairs), so)
    return X.with_columns(src_f=pl.col("src").cast(pl.Float32))


def evaluate(oof, truth_idx, by_c, thr):
    res = {}
    for c, ids in by_c.items():
        t = thr[c]
        sub = oof.filter(pl.col("i1").is_in(pl.Series(ids, dtype=pl.UInt32).implode()))
        pred = decode(sub.select(*KEYS, "p"), t["t_single"], t["t_match"], t["miss_mass"])
        res[c] = float(np.mean([f05(pred.get(i, ()), truth_idx[i]) for i in ids]))
    return res


def tune(oof, truth_idx, by_c, base, wide=False):
    """Neighbourhood grid around the base thresholds (same grid as metablock.main)."""
    out = {}
    for c, ids in by_c.items():
        sub = oof.filter(pl.col("i1").is_in(pl.Series(ids, dtype=pl.UInt32).implode())).select(*KEYS, "p")
        b, best = base[c], None
        ts_d = (-0.1, -0.05, 0, 0.05) if wide else (-0.05, 0, 0.05)
        for ts in sorted({round(min(0.97, max(0.5, b["t_single"] + d)), 2) for d in ts_d}):
            for tm in sorted({round(min(0.9, max(0.1, b["t_match"] + d)), 2) for d in (-0.1, 0, 0.1)}):
                if tm > ts:
                    continue
                for mm in sorted({round(b["miss_mass"] * f, 2) for f in (0.5, 1, 1.5)}):
                    pred = decode(sub, ts, tm, mm)
                    f = float(np.mean([f05(pred.get(i, ()), truth_idx[i]) for i in ids]))
                    if best is None or f > best[0]:
                        best = (f, ts, tm, mm)
        f, ts, tm, mm = best
        out[c] = dict(t_single=ts, t_match=tm, miss_mass=mm, oof_pruned_f05=round(f, 5))
        log(f"  {c}: F0.5={f:.5f} with t_single={ts} t_match={tm} miss_mass={mm}")
    out["_default"] = {k: max(out[c][k] for c in by_c) for k in ("t_single", "t_match", "miss_mass")}
    return out


def weighted(res):
    return sum(WEIGHTS[c] * res[c] for c in WEIGHTS if c in res) / sum(WEIGHTS[c] for c in WEIGHTS if c in res)


def main(sample_s1=None, write=True, second=None, baseline=None, suffix="_stack", wide=False, primary="mean",
         fixed_thr=None, min_gain=MIN_GAIN, iterate=False, unseen=None, rounds=ROUNDS, leaves=None, mini_tune=False):
    params = dict(PARAMS, num_leaves=leaves) if leaves else PARAMS
    s1 = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).with_row_index("i1")
    so = other_records("train")
    folds = np.random.default_rng(config.SEED + 1).integers(0, 2, s1.height).astype(np.int8)  # as train_oof.py
    sample = None
    if sample_s1:
        sample = pl.Series("i1", np.random.default_rng(7).choice(s1.height, sample_s1, replace=False).astype(np.uint32))

    oof = pl.scan_parquet(config.artifact("oof_train", "parquet")).select(*KEYS, "p")
    if sample is not None:
        oof = oof.filter(pl.col("i1").is_in(sample.implode()))
    oof = oof.collect().join(kept_train(sample), on=KEYS).with_columns(pl.col("p").cast(pl.Float32))
    feats = list(FEATS)
    if second:
        oof = blend(oof, pl.read_parquet(config.artifact(second, "parquet")), primary)
        feats += ["p_first", "p_second"]
    truth = blocking._truth_pairs("train", None)
    if sample is not None:
        truth = truth.filter(pl.col("i1").is_in(sample.implode()))
    fold_of = pl.DataFrame({"i1": np.arange(s1.height, dtype=np.uint32), "fold": folds})
    oof = (oof.join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left")
           .with_columns(pl.col("y").fill_null(0)).join(fold_of, on="i1"))
    ids = sample.to_list() if sample is not None else list(range(s1.height))
    truth_idx = {i: set() for i in ids}
    for i1, src, i2 in truth.iter_rows():
        truth_idx[i1].add((src, i2))
    country = s1["country"].to_numpy()
    by_c = {}
    for i in ids:
        by_c.setdefault(country[i], []).append(i)
    base_thr = json.loads(config.artifact("thresholds_final", "json").read_text())
    base = evaluate(oof, truth_idx, by_c, base_thr)
    log(f"pruned OOF: {oof.height} pairs; base F0.5 {base} weighted {weighted(base):.5f}")

    X = features(oof, so)
    log(f"features built: {X.height} rows x {len(feats)}")
    p2 = np.zeros(X.height, np.float32)
    boosters = []
    fold = X["fold"].to_numpy()
    for f in (0, 1):
        tr = X.filter(pl.Series(fold != f))
        booster = lgb.train(params, lgb.Dataset(tr.select(feats).cast(pl.Float32).to_numpy(),
                                                label=tr["y"].to_numpy(), feature_name=feats), rounds)
        p2[fold == f] = booster.predict(X.filter(pl.Series(fold == f)).select(feats).cast(pl.Float32).to_numpy())
        boosters.append(booster)
        log(f"stacker fold {f} done")
    imp = sorted(zip(boosters[0].feature_importance("gain"), feats), reverse=True)
    log("stacker top features: " + ", ".join(f"{n} {g / sum(g for g, _ in imp):.2f}" for g, n in imp[:8]))

    new_oof = X.select(*KEYS, p=pl.Series(p2, dtype=pl.Float32))
    if fixed_thr:  # thresholds already tuned by an earlier identical run
        thr = json.loads(Path(fixed_thr).read_text())
        new = evaluate(new_oof, truth_idx, by_c, thr)
        log(f"fixed thresholds from {fixed_thr}: {new} weighted {weighted(new):.5f}")
        if mini_tune:  # t_single +-0.05 around the fixed values
            for d in (-0.05, 0.05):
                t_try = dict(thr, **{c: dict(thr[c], t_single=round(thr[c]["t_single"] + d, 2)) for c in by_c})
                r = evaluate(new_oof, truth_idx, by_c, t_try)
                log(f"t_single {d:+.2f}: {r} weighted {weighted(r):.5f}")
                if weighted(r) > weighted(new):
                    new, thr = r, t_try
    else:
        log("re-tuning thresholds on the stacked pruned OOF")
        thr = tune(new_oof, truth_idx, by_c, base_thr, wide)
        new = {c: thr[c]["oof_pruned_f05"] for c in by_c}
    if unseen:
        thr["_default"] = dict(unseen)
    boosters2 = None
    if iterate:  # iterative collective classification: rebuild the neighbour features from round-1 p
        keep = [c for c in ("p_first", "p_second") if c in X.columns]
        R2 = features(X.select(*KEYS, "y", "fold", *keep, p_r1=pl.col("p"), p=pl.Series(p2, dtype=pl.Float32)), so)
        feats2 = feats + ["p_r1"]
        p3 = np.zeros(R2.height, np.float32)
        fold2 = R2["fold"].to_numpy()
        boosters2 = []
        for f in (0, 1):
            tr = R2.filter(pl.Series(fold2 != f))
            b2 = lgb.train(PARAMS, lgb.Dataset(tr.select(feats2).cast(pl.Float32).to_numpy(), label=tr["y"].to_numpy(),
                                               feature_name=feats2), ROUNDS)
            p3[fold2 == f] = b2.predict(R2.filter(pl.Series(fold2 == f)).select(feats2).cast(pl.Float32).to_numpy())
            boosters2.append(b2)
        oof2 = R2.select(*KEYS, p=pl.Series(p3, dtype=pl.Float32))
        del R2
        best2 = None
        for ts in (-0.05, 0.0, 0.05):  # small re-tune of t_single only
            t2 = {c: dict(thr[c], t_single=round(thr[c]["t_single"] + ts, 2)) for c in by_c}
            r = evaluate(oof2, truth_idx, by_c, t2)
            log(f"round 2, t_single {ts:+.2f}: {r} weighted {weighted(r):.5f}")
            if best2 is None or weighted(r) > weighted(best2[0]):
                best2 = (r, t2)
        if weighted(best2[0]) > weighted(new):
            log(f"round 2 adopted: {weighted(new):.5f} -> {weighted(best2[0]):.5f}")
            new = best2[0]
            thr = dict(thr, **best2[1])
        else:
            log("round 2 not better; keeping round 1")
            boosters2 = None
    ref = baseline if baseline is not None else weighted(base)  # e.g. the adopted stacker's score
    gain = weighted(new) - ref
    report = dict(base=base, new=new, weighted_base=weighted(base), reference=ref, weighted_new=weighted(new),
                  gain=gain, min_gain=min_gain, features=feats, second=second, rounds=rounds, leaves=leaves,
                  round2=boosters2 is not None, thresholds=thr)
    log(f"weighted F0.5 reference {ref:.5f} -> {weighted(new):.5f} (gain {gain:+.5f}; gate {min_gain})")
    out_dir = Path(f"{config.OUT_DIR}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stack_report.json").write_text(json.dumps(report, indent=2))
    if gain < min_gain or not write or sample is not None:
        log("not writing a submission (gain below gate, or sample / dry run)")
        return report

    # test: same features on the pruned, first-stage-scored test pairs
    del X, oof, new_oof
    t1 = pl.read_parquet(config.norm_path("test", 1), columns=["id", "country"]).with_row_index("i1")
    t2 = other_records("test")
    test = pl.read_parquet(config.artifact("scored_test", "parquet")).select(*KEYS, "p").with_columns(pl.col("p").cast(pl.Float32))
    if second:
        test = blend(test, pl.read_parquet(config.artifact(second.replace("oof_", "scored_test_"), "parquet")), primary)
    Xt = features(test, t2)
    Pt = np.mean([b.predict(Xt.select(feats).cast(pl.Float32).to_numpy()) for b in boosters], axis=0)
    if boosters2:
        keep = [c for c in ("p_first", "p_second") if c in Xt.columns]
        Xt = features(Xt.select(*KEYS, *keep, p_r1=pl.col("p"), p=pl.Series(Pt, dtype=pl.Float32)), t2)
        Pt = np.mean([b.predict(Xt.select(feats + ["p_r1"]).cast(pl.Float32).to_numpy()) for b in boosters2], axis=0)
    scored = Xt.select(*KEYS, p=pl.Series(Pt, dtype=pl.Float32))
    ids2 = {k: t2.filter(pl.col("src") == k).sort("i2")["id"].to_list() for k in (2, 3)}
    s1_ids = t1["id"].to_list()
    pred = {}
    for c in t1["country"].unique().to_list():
        t = thr.get(c, thr["_default"])
        pred.update(decode(scored.filter(pl.col("i1").is_in(t1.filter(pl.col("country") == c)["i1"].implode())),
                           t["t_single"], t["t_match"], t["miss_mass"]))
    matches = {s1_ids[i1]: [ids2[src][i2] for src, i2 in lst] for i1, lst in pred.items()}
    cand = scored.join(t2.select("src", "i2", "id"), on=["src", "i2"]).group_by("i1").agg(pl.col("id"))
    candidates = {s1_ids[i1]: lst for i1, lst in cand.iter_rows()}
    write_id_lists(out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, matches)
    write_id_lists(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, candidates)
    (out_dir / "thresholds_stack.json").write_text(json.dumps(thr, indent=2))
    for i, b in enumerate(boosters):
        b.save_model(str(config.artifact(f"model{suffix}{i}", "txt")))
    for i, b in enumerate(boosters2 or []):
        b.save_model(str(config.artifact(f"model{suffix}_r2_{i}", "txt")))
    n_match = sum(len(v) for v in matches.values())
    log(f"wrote {out_dir}: {len(matches)} of {len(s1_ids)} S1 matched, {n_match / len(s1_ids):.2f} links per S1, "
        f"{scored.height / len(s1_ids):.2f} candidates per S1")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-s1", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--second", default=None, help="artifact of a second first-stage OOF, e.g. oof_pruned")
    ap.add_argument("--baseline", type=float, default=None, help="weighted F0.5 to beat (default: first stage)")
    ap.add_argument("--suffix", default="_stack")
    ap.add_argument("--wide-grid", action="store_true", help="also try t_single 0.1 below the base")
    ap.add_argument("--primary", default="mean", choices=["mean", "second"], help="p used for decoding context")
    ap.add_argument("--fixed-thr", default=None, help="thresholds json from an earlier identical run (skips tuning)")
    ap.add_argument("--min-gain", type=float, default=MIN_GAIN)
    ap.add_argument("--iterate", action="store_true", help="second collective round on round-1 probabilities")
    ap.add_argument("--strict-unseen", action="store_true", help="countries without labels (France) use UNSEEN")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--leaves", type=int, default=None)
    ap.add_argument("--mini-tune", action="store_true")
    a = ap.parse_args()
    main(a.sample_s1, write=not a.dry_run, second=a.second, baseline=a.baseline, suffix=a.suffix, wide=a.wide_grid, primary=a.primary,
         fixed_thr=a.fixed_thr, min_gain=a.min_gain, iterate=a.iterate,
         unseen=UNSEEN if a.strict_unseen else None, rounds=a.rounds, leaves=a.leaves, mini_tune=a.mini_tune)
