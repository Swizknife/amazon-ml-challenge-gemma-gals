"""IDF-weighted overlap features (v10).

A shared rare word ("Beloumbra", "Eksar") is strong evidence and a shared common word
("traders", "road") is weak, but token-set ratios count them alike. Team Magnum (public LB
0.9848) reported IDF-weighted name overlap as their single best feature (+0.010). Per pair,
with idf(t) = log(N / df(t)) over each country's records of the split (S1 + S2 + S3, the
split's own text, like every other token statistic here):

  idf_name_jac   sum idf(shared core-name tokens) / sum idf(union)
  idf_name_cov1  ... / sum idf(S1 tokens)       idf_name_cov2  ... / sum idf(candidate tokens)
  idf_addr_jac, idf_addr_cov1, idf_addr_cov2     the same over address words
  idf_name_miss  idf of the rarest name token one side has and the other lacks
A side with no tokens leaves its features null (unknown, not zero).

    python featidf.py          # writes feats_idf_train.parquet (kept training pairs)
    featidf.compute(split, keys) -> frame (test features are built on the fly by train_pruned.py)
"""
import time

import polars as pl

import config
import features
from features import KEYS

L = pl.col
NEW = ["idf_name_jac", "idf_name_cov1", "idf_name_cov2", "idf_addr_jac", "idf_addr_cov1", "idf_addr_cov2",
       "idf_name_miss"]
T0 = time.time()


def log(msg):
    print(f"[featidf {time.time() - T0:6.0f}s] {msg}", flush=True)


def records(split):
    cols = ["country", "core", "ad"]
    s1 = pl.read_parquet(config.norm_path(split, 1), columns=cols).with_row_index("i1")
    so = pl.concat([pl.read_parquet(config.norm_path(split, k), columns=cols).with_row_index("i2")
                    .with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    s1 = s1.with_columns(ct=features._toks("core"), at=features._toks("ad"))
    so = so.with_columns(ct=features._toks("core"), at=features._toks("ad"))
    idf = {}
    for col in ("ct", "at"):
        both = pl.concat([s1.select("country", t=col), so.select("country", t=col)])
        n = both.group_by("country").len().rename({"len": "n"})
        idf[col] = (both.explode("t").drop_nulls().group_by("country", "t").len()
                    .join(n, on="country").select("country", "t", idf=(L("n") / L("len")).log().cast(pl.Float32)))
    return s1.drop("core", "ad"), so.drop("core", "ad"), idf


def _overlap(ch, col, idf, prefix):
    """Weighted Jaccard / coverage of one token list between the two sides of each pair."""
    a = ch.select("row", "country", t=f"{col}_1").explode("t").drop_nulls().join(idf, on=["country", "t"])
    b = ch.select("row", "country", t=f"{col}_2").explode("t").drop_nulls().join(idf, on=["country", "t"])
    sa = a.group_by("row").agg(sa=L("idf").sum())
    sb = b.group_by("row").agg(sb=L("idf").sum())
    sh = a.join(b.select("row", "t"), on=["row", "t"]).group_by("row").agg(sh=L("idf").sum())
    out = (ch.select("row").join(sa, on="row", how="left").join(sb, on="row", how="left")
           .join(sh, on="row", how="left").with_columns(L("sh").fill_null(0.0)))
    known = L("sa").is_not_null() & L("sb").is_not_null()
    cols = {f"{prefix}_jac": pl.when(known).then(L("sh") / (L("sa") + L("sb") - L("sh")).clip(1e-6)),
            f"{prefix}_cov1": pl.when(known).then(L("sh") / L("sa").clip(1e-6)),
            f"{prefix}_cov2": pl.when(known).then(L("sh") / L("sb").clip(1e-6))}
    if prefix == "idf_name":  # the most informative disagreeing token
        miss = (pl.concat([a.join(b.select("row", "t"), on=["row", "t"], how="anti"),
                           b.join(a.select("row", "t"), on=["row", "t"], how="anti")])
                .group_by("row").agg(idf_name_miss=L("idf").max()))
        out = out.join(miss, on="row", how="left")
        cols["idf_name_miss"] = pl.when(known).then(L("idf_name_miss").fill_null(0.0))  # 0: nothing disagrees
    return out.sort("row").select(**{k: v.cast(pl.Float32) for k, v in cols.items()})


def pair_features(keys, s1, so, idf, chunk=2_000_000):
    r1 = s1.select("i1", "country", ct_1="ct", at_1="at")
    r2 = so.select("src", "i2", ct_2="ct", at_2="at")
    out = []
    for start in range(0, keys.height, chunk):
        ch = (keys.slice(start, chunk).select(KEYS).with_row_index("row")
              .join(r1, on="i1", how="left").join(r2, on=["src", "i2"], how="left").sort("row"))
        out.append(pl.concat([ch.select(KEYS), _overlap(ch, "ct", idf["ct"], "idf_name"),
                              _overlap(ch, "at", idf["at"], "idf_addr")], how="horizontal"))
    return pl.concat(out).select(*KEYS, *NEW)


def compute(split, keys):
    s1, so, idf = records(split)
    return pair_features(keys, s1, so, idf)


def main():
    feats = compute("train", pl.read_parquet(config.artifact("kept_train", "parquet")))
    feats.write_parquet(config.artifact("feats_idf_train", "parquet"))
    log(f"train: {feats.height} pairs -> feats_idf_train")


if __name__ == "__main__":
    main()
