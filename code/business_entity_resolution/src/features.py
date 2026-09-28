"""Pair features for (S1, candidate) pairs, computed chunk by chunk.

Three groups:
  string   rapidfuzz similarities of names / addresses (vectorized, multi-threaded)
  sets     token, number and legal-form agreement via polars list operations
  context  how this pair ranks among the S1's candidates and among all S1s that
           compete for the same candidate ("rival" signals); from the blocking
           pre-score only, so no model output leaks into features
No country indicator is used, so the model can transfer to unseen countries.
"""
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

import config
import rules

REC_COLS = ["country", "nm", "core", "legal", "cc", "script", "web", "ad", "adf", "nums", "hn", "hmain"]
_GENERIC = sorted(rules.ADDR_GENERIC)


def _toks(col):
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def load_records(split):
    """S1 records (index i1) and S2+S3 records (index src, i2) with token lists,
    each record's rarest name token / address word, and their document frequency."""
    s1 = pl.read_parquet(config.norm_path(split, 1)).select("id", *REC_COLS).with_row_index("i1")
    so = pl.concat([pl.read_parquet(config.norm_path(split, k)).select("id", *REC_COLS)
                    .with_row_index("i2").with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    s1 = s1.with_columns(ct=_toks("core"), at=_toks("ad"), nt=_toks("nums"))
    so = so.with_columns(ct=_toks("core"), at=_toks("ad"), nt=_toks("nums"))
    both = pl.concat([s1.select("ct", "at"), so.select("ct", "at")])
    dfc = both.select("ct").explode("ct").drop_nulls().group_by("ct").len().rename({"ct": "t", "len": "df"})
    dfa = (both.select("at").explode("at").drop_nulls().filter(~pl.col("at").is_in(_GENERIC))
           .group_by("at").len().rename({"at": "t", "len": "df"}))
    s1 = _add_rarest(_add_rarest(s1, "i1", "ct", dfc, "rc"), "i1", "at", dfa, "ra")
    so = _add_rarest(_add_rarest(so, ["src", "i2"], "ct", dfc, "rc"), ["src", "i2"], "at", dfa, "ra")
    return s1, so


def _add_rarest(f, key, col, dfreq, name):
    key = [key] if isinstance(key, str) else key
    r = (f.select(*key, t=pl.col(col)).explode("t").drop_nulls().join(dfreq, on="t")
         .sort([*key, "df", "t"]).group_by(key, maintain_order=True).first()
         .rename({"t": name, "df": f"{name}_df"}))
    return f.join(r, on=key, how="left").with_columns(pl.col(f"{name}_df").fill_null(0))


def context(cands):
    """Rank / relative-score features over the whole candidate universe, plus the
    S1's best other candidate (`ref_src`, `ref_i2`) used for consistency features."""
    cands = cands.with_columns(
        rank1=pl.col("pre").rank("ordinal", descending=True).over("i1").cast(pl.Float32),
        rank1s=pl.col("pre").rank("ordinal", descending=True).over("i1", "src").cast(pl.Float32),
        rel1=(pl.col("pre") / pl.col("pre").max().over("i1")).cast(pl.Float32),
        rank2=pl.col("pre").rank("ordinal", descending=True).over("src", "i2").cast(pl.Float32),
        rel2=(pl.col("pre") / pl.col("pre").max().over("src", "i2")).cast(pl.Float32),
        n2=pl.len().over("src", "i2").cast(pl.Float32),
    )
    top = cands.filter(pl.col("rank1") <= 2).select("i1", "rank1", "src", "i2")
    t1 = top.filter(pl.col("rank1") == 1).select("i1", t1s="src", t1i="i2")
    t2 = top.filter(pl.col("rank1") == 2).select("i1", t2s="src", t2i="i2")
    is_top = (pl.col("src") == pl.col("t1s")) & (pl.col("i2") == pl.col("t1i"))
    return (cands.join(t1, on="i1", how="left").join(t2, on="i1", how="left")
            .with_columns(ref_src=pl.when(is_top).then("t2s").otherwise("t1s"),
                          ref_i2=pl.when(is_top).then("t2i").otherwise("t1i"))
            .drop("t1s", "t1i", "t2s", "t2i"))


def _fuzzy(ch):
    out = {}
    specs = [("c_ratio", "core", fuzz.ratio), ("c_tset", "core", fuzz.token_set_ratio),
             ("c_tsort", "core", fuzz.token_sort_ratio), ("c_part", "core", fuzz.partial_ratio),
             ("c_jw", "core", distance.JaroWinkler.normalized_similarity),
             ("cc_ratio", "cc", fuzz.ratio), ("n_tset", "nm", fuzz.token_set_ratio),
             ("a_tset", "adf", fuzz.token_set_ratio), ("a_ratio", "adf", fuzz.ratio),
             ("w_tset", "ad", fuzz.token_set_ratio), ("w_part", "ad", fuzz.partial_ratio)]
    cache = {}
    for name, col, scorer in specs:
        if col not in cache:
            cache[col] = (ch[f"{col}_1"].to_list(), ch[f"{col}_2"].to_list())
        a, b = cache[col]
        out[name] = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    return pl.DataFrame(out)


def pair_features(ch):
    """ch: candidate rows joined with record columns suffixed _1 (S1) / _2 (other)."""
    L = lambda c: pl.col(c)
    sets = ch.select(
        c_inter=L("ct_1").list.set_intersection(L("ct_2")).list.len(),
        c_union=L("ct_1").list.set_union(L("ct_2")).list.len(),
        c_only1=L("ct_1").list.set_difference(L("ct_2")).list.len(),
        c_only2=L("ct_2").list.set_difference(L("ct_1")).list.len(),
        a_inter=L("at_1").list.set_intersection(L("at_2")).list.len(),
        a_union=L("at_1").list.set_union(L("at_2")).list.len(),
        n_inter=L("nt_1").list.set_intersection(L("nt_2")).list.len(),
        n_len1=L("nt_1").list.len(), n_len2=L("nt_2").list.len(),
        hmain_eq=(L("hmain_1") == L("hmain_2")) & (L("hmain_1") != ""),
        hn_eq=(L("hn_1") == L("hn_2")) & (L("hn_1") != ""),
        h1_in_2=L("nt_2").list.contains(L("hmain_1")),
        h2_in_1=L("nt_1").list.contains(L("hmain_2")),
        hdiff=(L("hmain_1").cast(pl.Float64, strict=False) - L("hmain_2").cast(pl.Float64, strict=False)).abs().log1p(),
        legal_eq=L("legal_1") == L("legal_2"),
        legal_e1=L("legal_1") == "", legal_e2=L("legal_2") == "",
        rc1_in_2=L("ct_2").list.contains(L("rc_1")), rc2_in_1=L("ct_1").list.contains(L("rc_2")),
        rc_df1=L("rc_df_1").log1p(), rc_df2=L("rc_df_2").log1p(),
        ra1_in_2=L("at_2").list.contains(L("ra_1")), ra2_in_1=L("at_1").list.contains(L("ra_2")),
        ra_df1=L("ra_df_1").log1p(), ra_df2=L("ra_df_2").log1p(),
        cc_in=L("cc_2").str.contains(L("cc_1"), literal=True) | L("cc_1").str.contains(L("cc_2"), literal=True),
        script_1=L("script_1"), script_2=L("script_2"), web_2=L("web_2"),
        addr_empty2=(L("ad_2") == "") & (L("nums_2") == ""),
        len_c1=L("ct_1").list.len(), len_c2=L("ct_2").list.len(),
    ).with_columns(
        c_jac=pl.col("c_inter") / pl.col("c_union").clip(1),
        a_jac=pl.col("a_inter") / pl.col("a_union").clip(1),
        n_jac=pl.col("n_inter") / (pl.col("n_len1") + pl.col("n_len2") - pl.col("n_inter")).clip(1),
    )
    sets = sets.with_columns(pl.col(pl.Boolean).cast(pl.Int8))
    return pl.concat([sets, _fuzzy(ch)], how="horizontal")


KEYS = ["i1", "src", "i2"]
CTX = ["pre", "nk", "rank1", "rank1s", "rel1", "rank2", "rel2", "n2"]
KIND_BITS = {"nw": 1, "np": 2, "nt": 4, "cc": 8, "ap": 16, "nn": 32, "hw": 64, "nr": 128, "em": 256}  # mirrors blocking.KIND_BITS
W_COLS = [f"w_{k}" for k in KIND_BITS if k != "em"]  # per-key-kind evidence (v4 blocking), when present


def _emb_cos(ch, emb):
    """Cosine of the stored name / address vectors for every pair of the chunk."""
    i1, src, i2 = ch["i1"].to_numpy(), ch["src"].to_numpy(), ch["i2"].to_numpy()
    cols = {}
    for what in ("name", "addr"):
        a = emb[what][1][i1].astype(np.float32)
        b = np.empty_like(a)
        for k in (2, 3):
            m = src == k
            b[m] = emb[what][k][i2[m]]
        cols[f"e_{what}_cos"] = (a * b).sum(axis=1)
    return pl.DataFrame(cols)


def build(cands, s1, so, chunk=1_500_000, emb=None):
    """Yield feature frames (keys + context + pair features) for `cands`, chunked."""
    r1 = s1.select("i1", *[pl.col(c).alias(f"{c}_1") for c in
                           ["core", "cc", "nm", "legal", "adf", "ad", "nums", "hn", "hmain", "script",
                            "ct", "at", "nt", "rc", "rc_df", "ra", "ra_df"]])
    r2 = so.select("src", "i2", *[pl.col(c).alias(f"{c}_2") for c in
                                  ["core", "cc", "nm", "legal", "adf", "ad", "nums", "hn", "hmain", "script",
                                   "web", "ct", "at", "nt", "rc", "rc_df", "ra", "ra_df"]])
    rr = so.select(ref_src="src", ref_i2="i2", core_r="core", adf_r="adf", hn_r="hn")
    for start in range(0, cands.height, chunk):
        ch = (cands.slice(start, chunk).join(r1, on="i1", how="left").join(r2, on=["src", "i2"], how="left")
              .join(rr, on=["ref_src", "ref_i2"], how="left")
              .with_columns(pl.col("core_r", "adf_r", "hn_r").fill_null("")))
        feats = pair_features(ch)
        cons = pl.DataFrame({
            "r_c_tset": cpdist(ch["core_2"].to_list(), ch["core_r"].to_list(),
                               scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
            "r_a_tset": cpdist(ch["adf_2"].to_list(), ch["adf_r"].to_list(),
                               scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
        }).with_columns(
            r_hn_eq=((ch["hn_2"] == ch["hn_r"]) & (ch["hn_2"] != "")).cast(pl.Int8),
            has_ref=(ch["core_r"] != "").cast(pl.Int8))
        parts = [ch.select(*KEYS, *CTX, *[c for c in W_COLS if c in ch.columns]), feats, cons]
        if "kinds" in ch.columns:  # which blocking key kinds the pair shared
            bits = KIND_BITS if emb is not None else {k: v for k, v in KIND_BITS.items() if k != "em"}
            parts.append(ch.select(**{f"k_{name}": (pl.col("kinds") & bit > 0).cast(pl.Int8)
                                      for name, bit in bits.items()}))
        if emb is not None:
            parts.append(_emb_cos(ch, emb))
        yield pl.concat(parts, how="horizontal")


FEATURES = None  # filled on first use: every non-key column of a built frame


def feature_names(frame):
    return [c for c in frame.columns if c not in KEYS and c != "y"]
