"""Export the uncertain pairs for the Kaggle cross-encoder (src/kaggle/xenc_kaggle.py).

Band: band < p < 1 - band on the second matcher's probability -- out-of-fold on train
(oof_pruned<tag>), test from scored_test_pruned<tag> -- the same definition on both sides.
Columns: i1, src, i2, p, fold (the train_oof.py entity folds; test: -1), y (train only),
t1 / t2 = raw "name | address" of the S1 record and of the candidate.
The data is the challenge's own records: keep the Kaggle dataset private.

    python xenc_export.py [--tag _v10] [--band 0.02] [--out <dir>]
Writes <out>/xenc_train.parquet, <out>/xenc_test.parquet
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

import blocking
import config
from xenc import records

KEYS = ["i1", "src", "i2"]
T0 = time.time()


def log(msg):
    print(f"[export {time.time() - T0:6.0f}s] {msg}", flush=True)


def band_pairs(name, band):
    P = pl.read_parquet(config.artifact(name, "parquet"), columns=[*KEYS, "p"])
    return P.filter((pl.col("p") > band) & (pl.col("p") < 1 - band))


def export(tag="", band=0.02, out=None):
    out = Path(out or config.WORK_DIR / "kaggle_data")
    out.mkdir(parents=True, exist_ok=True)
    s1, so = records("train")
    folds = np.random.default_rng(config.SEED + 1).integers(0, 2, s1.height).astype(np.int8)  # as train_oof.py
    P = band_pairs(f"oof_pruned{tag}", band)
    truth = blocking._truth_pairs("train", None)
    tr = (P.join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left").with_columns(pl.col("y").fill_null(0))
          .join(pl.DataFrame({"i1": np.arange(s1.height, dtype=np.uint32), "fold": folds}), on="i1")
          .join(s1.select("i1", "t1"), on="i1").join(so.select("src", "i2", "t2"), on=["src", "i2"]))
    tr.write_parquet(out / "xenc_train.parquet", compression="zstd")
    log(f"train band pairs: {tr.height} (positive rate {tr['y'].mean():.3f}) -> {out / 'xenc_train.parquet'}")
    t1, t2 = records("test")
    te = (band_pairs(f"scored_test_pruned{tag}", band).with_columns(fold=pl.lit(-1, pl.Int8))
          .join(t1.select("i1", "t1"), on="i1").join(t2.select("src", "i2", "t2"), on=["src", "i2"]))
    te.write_parquet(out / "xenc_test.parquet", compression="zstd")
    log(f"test band pairs: {te.height} -> {out / 'xenc_test.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--band", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    export(a.tag, a.band, a.out)
