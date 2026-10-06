"""Context counts: global statistics a pairwise matcher cannot infer by itself.

The 2 Oct calibration check (evaluate.py) found the matcher blind to one number: how many S1
businesses share a name. On empty-address, same-name pairs it under-predicted unique names
by 0.033 and over-predicted names shared by 2-5 S1s by up to 0.029, while the blocking rival
count n2 was calibrated. Counting belongs to code, not to the model (Deng et al., arXiv
2609.24965: the model makes the semantic choice, deterministic code does the arithmetic), so
the counts become features. Buckets 1 / 2 / 3-5 / 6+ (coded 1-4) keep them comparable between
train and test, whose densities differ.

  S1 side   a_s1_name       S1s in the country with the same normalized core name
            a_s1_bldg       S1s at the same building key (main house number + rarest address word)
            a_s1_name_bldg  S1s sharing both the name and the building (siblings)
  pair      core_eq         identical normalized core names
            cc_eq           identical space-free names (spacing / website variants)
  records   a_rec_name, a_rec_name_noaddr   (only with --records)
            S2+S3 records in the country with the same core name, all and empty-address only.
            Test has 24% more records per S1 than train, so these can shift: they are opt-in.
Counts come from each split's own text (unsupervised statistics, like the token frequencies).

    python featctx.py [--records]   # writes feats_ctx_train.parquet and feats_ctx_test.parquet
                                    # and prints the train/test bucket shares (shift check)
"""
import argparse
import time

import polars as pl

import config
import features
from features import KEYS

L = pl.col
S1_COLS = ["a_s1_name", "a_s1_bldg", "a_s1_name_bldg"]
REC_COLS = ["a_rec_name", "a_rec_name_noaddr"]
T0 = time.time()


def log(msg):
    print(f"[featctx {time.time() - T0:6.0f}s] {msg}", flush=True)


def bucket(e):
    """0, 1, 2, 3-5 -> 3, 6+ -> 4; null (no name / no building key) stays null."""
    return (pl.when(e.is_null()).then(None).when(e <= 2).then(e).when(e <= 5).then(3)
            .otherwise(4)).cast(pl.Int8)


def records(split):
    cols = ["country", "core", "cc", "ad", "nums", "hmain"]
    s1 = pl.read_parquet(config.norm_path(split, 1), columns=cols).with_row_index("i1")
    so = pl.concat([pl.read_parquet(config.norm_path(split, k), columns=cols).with_row_index("i2")
                    .with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    # rarest address word, with the same document frequencies as features.load_records
    at = pl.concat([s1.select(at=features._toks("ad")), so.select(at=features._toks("ad"))])
    dfa = (at.explode("at").drop_nulls().filter(~L("at").is_in(features._GENERIC))
           .group_by("at").len().rename({"at": "t", "len": "df"}))
    s1 = features._add_rarest(s1.with_columns(at=features._toks("ad")), "i1", "at", dfa, "ra").drop("at")
    return s1_counts(s1), rec_counts(so)


def s1_counts(s1):
    """s1: country, core, hmain, ra (rarest address word)."""
    named, bldg = L("core") != "", (L("hmain") != "") & L("ra").is_not_null()
    return s1.with_columns(
        a_s1_name=bucket(pl.when(named).then(pl.len().over("country", "core"))),
        a_s1_bldg=bucket(pl.when(bldg).then(pl.len().over("country", "hmain", "ra"))),
        a_s1_name_bldg=bucket(pl.when(named & bldg).then(pl.len().over("country", "core", "hmain", "ra"))))


def rec_counts(so):
    """so: country, core, ad, nums."""
    named = L("core") != ""
    empty = ((L("ad") == "") & (L("nums") == "")).cast(pl.UInt32)
    return so.with_columns(
        a_rec_name=bucket(pl.when(named).then(pl.len().over("country", "core"))),
        a_rec_name_noaddr=bucket(pl.when(named).then(empty.sum().over("country", "core"))))


def pair_features(keys, s1, so, with_records=False):
    r1 = s1.select("i1", *S1_COLS, core_1="core", cc_1="cc")
    r2 = so.select("src", "i2", *(REC_COLS if with_records else []), core_2="core", cc_2="cc")
    X = keys.select(KEYS).join(r1, on="i1", how="left").join(r2, on=["src", "i2"], how="left")
    return X.select(*KEYS, *S1_COLS, *(REC_COLS if with_records else []),
                    core_eq=((L("core_1") == L("core_2")) & (L("core_1") != "")).cast(pl.Int8),
                    cc_eq=((L("cc_1") == L("cc_2")) & (L("cc_1") != "")).cast(pl.Int8))


def shares(s1, so, split):
    """Bucket shares per country, for the train/test shift check."""
    rows = []
    for col, f in [(c, s1) for c in S1_COLS] + [(c, so) for c in REC_COLS]:
        t = (f.group_by("country", col).len().with_columns(share=L("len") / L("len").sum().over("country"))
             .select(split=pl.lit(split), feature=pl.lit(col), country="country", bucket=L(col), share="share"))
        rows.append(t)
    return pl.concat(rows)


def compute(split, keys, with_records=False):
    """Features for any pair set of a split (train_pruned.py builds test features this way)."""
    s1, so = records(split)
    return pair_features(keys, s1, so, with_records)


def build(split, keys, with_records=False):
    s1, so = records(split)
    feats = pair_features(keys, s1, so, with_records)
    feats.write_parquet(config.artifact(f"feats_ctx_{split}", "parquet"))
    log(f"{split}: {feats.height} pairs -> feats_ctx_{split} ({feats.width - 3} features)")
    return shares(s1, so, split)


def main(with_records=False):
    tr = build("train", pl.read_parquet(config.artifact("kept_train", "parquet")), with_records)
    te = build("test", pl.read_parquet(config.artifact("scored_test_pruned", "parquet"), columns=KEYS), with_records)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(80)
    cmp = (tr.join(te, on=["feature", "country", "bucket"], how="full", coalesce=True, suffix="_test")
           .select("feature", "country", "bucket", train=L("share").round(4), test=L("share_test").round(4))
           .sort("feature", "country", "bucket"))
    print("== bucket shares, train vs test (shift check; France has no train rows)")
    print(cmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", action="store_true", help="also write the record-side counts (shift-prone)")
    main(ap.parse_args().records)
