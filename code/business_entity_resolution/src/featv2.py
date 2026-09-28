"""Look-alike (sibling-trap) features, v2: the details normalization throws away.

Our false merges are siblings in the same building. normalize.py drops unit words and
French bis/ter, and hn keeps digits only, so "17 BIS RUE X" == "17 RUE X",
"4233D LIBERTY RD" == "4233 LIBERTY RD" and "SHOP NO. G-4" vs "G-6" rest on one digit.
Measured on the sources: units appear in 20% of India addresses, bis/ter in 3.8% of
France addresses (≈0 in train), letter suffixes in 2.4% (US) / 10% (India).
New per-pair features, parsed from the raw address:
  units   u_n1, u_n2, u_eq, u_conf    shop / flat / unit / suite / office numbers and
                                      letter-dash codes (G-4); conflict = both have
                                      units and share none
  suffix  hs_1, hs_2, hs_eq, hs_conf  house-number letter suffix (4233D, 21 A); French
                                      bis / ter / quater map to b / c / d, so France
                                      reuses what train teaches about suffixes
  digits  h_lev, h_pre                edit distance of the main house numbers; one is a
                                      prefix of the other (272 vs 27: a dropped digit)
  extra   x1_df, x2_df, x1_in_a2, x2_in_a1  rarest extra name token per side (log df) and
                                      whether an extra name token is a word of the other
                                      record's address (a branch / location name)
Document frequencies come from each split's own text (test: unsupervised statistics only).

    python featv2.py            # writes feats_v2_train.parquet and feats_v2_test.parquet
Then compare with the baseline:
    python loco.py --extra feats_v2_train
    python train_pruned.py --extra feats_v2 --tag _v2
    python stack.py --second oof_pruned_v2 --primary second --wide-grid --strict-unseen
"""
import time

import numpy as np
import polars as pl
from rapidfuzz import distance
from rapidfuzz.process import cpdist

import config
from features import KEYS

UNIT_WORDS = r"(?:shop|shp|unit|suite|ste|apt|flat|office|off|room|rm|gala|stall|bat|batiment)"
UNIT_RE = UNIT_WORDS + r"\b\.?\s*(?:no\b\.?)?\s*[:#.\-]?\s*[a-z]{0,2}-?\d+[a-z]?\b|\b[a-z]{1,2}-\d+\b"
UNIT_HEAD = r"^" + UNIT_WORDS + r"\b\.?\s*(?:no\b\.?)?\s*[:#.\-]?\s*"
# first digit run of the address, an attached single letter (4233d) or a spaced suffix
# (21 a, 17 bis); spaced e/n/s/w are directions, so only a-d are taken there
SUFFIX_RE = r"(\d+)(?:([a-z])\b)?(?:\s+(bis|ter|quater|[a-d])\b)?"
FRENCH = {"bis": "b", "ter": "c", "quater": "d"}
NEW = ["u_n1", "u_n2", "u_eq", "u_conf", "hs_1", "hs_2", "hs_eq", "hs_conf", "h_lev", "h_pre",
       "x1_df", "x2_df", "x1_in_a2", "x2_in_a1"]
T0 = time.time()


def log(msg):
    print(f"[featv2 {time.time() - T0:6.0f}s] {msg}", flush=True)


def _toks(col):
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def _records(split, k):
    raw = (pl.col("business_address").fill_null("").str.to_lowercase().str.normalize("NFKD")
           .str.replace_all(r"\p{Mn}", ""))
    suf = raw.str.extract_groups(SUFFIX_RE)
    return (pl.read_parquet(config.norm_path(split, k), columns=["business_address", "core", "ad", "hmain"])
            .with_row_index("i2" if k > 1 else "i1")
            .with_columns(
                units=raw.str.extract_all(UNIT_RE).list.eval(
                    pl.element().str.replace(UNIT_HEAD, "").str.replace_all("-", "")
                    .str.replace(r"^([a-z]*)0+(\d)", "${1}${2}")).list.unique(),
                hsuf=pl.coalesce(suf.struct.field("2"), suf.struct.field("3")).replace(FRENCH).fill_null(""),
                ct=_toks("core"), at=_toks("ad"))
            .drop("business_address", "core", "ad"))


def records(split):
    s1 = _records(split, 1)
    so = pl.concat([_records(split, k).with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    df = (pl.concat([s1.select("ct"), so.select("ct")]).explode("ct").drop_nulls()
          .group_by("ct").len().select(t="ct", df=pl.col("len").log1p().cast(pl.Float32)))
    return s1, so, df


def _extra_df(ch, side, df):
    """log df of the rarest name token that only one side has (null when there is none)."""
    other = "2" if side == "1" else "1"
    ex = (ch.select("row", t=pl.col(f"ct_{side}").list.set_difference(pl.col(f"ct_{other}")))
          .explode("t").drop_nulls().join(df, on="t").group_by("row").agg(pl.col("df").min()))
    return ch.select("row").join(ex, on="row", how="left", maintain_order="left")["df"]


def pair_features(pairs, s1, so, df, chunk=2_000_000):
    r1 = s1.select("i1", *[pl.col(c).alias(f"{c}_1") for c in ("units", "hsuf", "hmain", "ct", "at")])
    r2 = so.select("src", "i2", *[pl.col(c).alias(f"{c}_2") for c in ("units", "hsuf", "hmain", "ct", "at")])
    out = []
    L = pl.col
    for start in range(0, pairs.height, chunk):
        ch = (pairs.slice(start, chunk).select(KEYS).join(r1, on="i1", how="left")
              .join(r2, on=["src", "i2"], how="left").with_row_index("row"))
        lev = np.array(cpdist(ch["hmain_1"].fill_null("").to_list(), ch["hmain_2"].fill_null("").to_list(),
                              scorer=distance.Levenshtein.distance, workers=-1, dtype=np.float32))
        lev[((ch["hmain_1"] == "") | (ch["hmain_2"] == "")).fill_null(True).to_numpy()] = np.nan
        f = ch.select(
            *KEYS,
            u_n1=L("units_1").list.len(), u_n2=L("units_2").list.len(),
            u_eq=L("units_1").list.set_intersection(L("units_2")).list.len() > 0,
            u_conf=(L("units_1").list.len() > 0) & (L("units_2").list.len() > 0)
            & (L("units_1").list.set_intersection(L("units_2")).list.len() == 0),
            hs_1=L("hsuf_1") != "", hs_2=L("hsuf_2") != "",
            hs_eq=L("hsuf_1") == L("hsuf_2"),
            hs_conf=(L("hmain_1") == L("hmain_2")) & (L("hmain_1") != "") & (L("hsuf_1") != L("hsuf_2")),
            h_pre=(L("hmain_1") != L("hmain_2")) & (L("hmain_1") != "") & (L("hmain_2") != "")
            & (L("hmain_1").str.starts_with(L("hmain_2")) | L("hmain_2").str.starts_with(L("hmain_1"))),
            x1_in_a2=L("ct_1").list.set_difference(L("ct_2")).list.set_intersection(L("at_2")).list.len() > 0,
            x2_in_a1=L("ct_2").list.set_difference(L("ct_1")).list.set_intersection(L("at_1")).list.len() > 0,
        ).with_columns(pl.col(pl.Boolean).cast(pl.Int8), pl.col(pl.UInt32).exclude(KEYS).cast(pl.Float32),
                       h_lev=pl.Series(lev), x1_df=_extra_df(ch, "1", df), x2_df=_extra_df(ch, "2", df))
        out.append(f.select(*KEYS, *NEW))
    return pl.concat(out)


def build(split, keys):
    s1, so, df = records(split)
    log(f"{split}: records parsed; S1 with units {s1['units'].list.len().gt(0).mean():.3f}, "
        f"with suffix {(s1['hsuf'] != '').mean():.3f}")
    feats = pair_features(keys, s1, so, df)
    feats.write_parquet(config.artifact(f"feats_v2_{split}", "parquet"))
    log(f"{split}: {feats.height} pairs -> feats_v2_{split}")
    return feats


def main():
    build("train", pl.read_parquet(config.artifact("kept_train", "parquet")))
    build("test", pl.read_parquet(config.artifact("scored_test_pruned", "parquet"), columns=KEYS))


if __name__ == "__main__":
    main()
