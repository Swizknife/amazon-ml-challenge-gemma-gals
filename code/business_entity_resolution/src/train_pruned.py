"""Second first-stage opinion: a matcher trained on the meta-blocked candidates it actually scores.

train.py fits on all ~28 blocking candidates of 400k entities. Here each of two folds of
Source-1 entities (the train_oof.py folds) fits LightGBM on the pruned candidates of its
~1.1M entities and scores the other fold, giving out-of-fold probabilities for every
pruned training pair. Test pairs are scored by the mean of the two fold models. stack.py
(--second oof_pruned) then uses both opinions (this one and train_oof.py's) as features.

    python train_pruned.py [--extra feats_v2 --tag _v2]
Writes kept_train.parquet, oof_pruned.parquet, scored_test_pruned.parquet, model_pruned{0,1}.txt
(--extra joins <extra>_train / <extra>_test feature artifacts, e.g. featv2.py's; --tag
suffixes the outputs: oof_pruned_v2.parquet, scored_test_pruned_v2.parquet, ...)
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import blocking
import config
import features
import metablock
from train import PARAMS

T0 = time.time()


def log(msg):
    print(f"[pruned {time.time() - T0:6.0f}s] {msg}", flush=True)


def kept_pairs():
    path = config.artifact("kept_train", "parquet")
    if not path.exists():
        metablock._kept_train(metablock.TAU).write_parquet(path)
    return pl.read_parquet(path)


def main(max_rounds=2000, extra=None, tag=""):
    kept = kept_pairs()
    log(f"kept training pairs: {kept.height}")
    parts = sorted((config.WORK_DIR / "feats_train").glob("*.parquet"))  # part by part: bounded memory
    X = pl.concat([pl.read_parquet(p).join(kept, on=features.KEYS) for p in parts])
    if extra:
        X = X.join(pl.read_parquet(config.artifact(f"{extra}_train", "parquet")), on=features.KEYS, how="left")
    names = [c for c in X.columns if c not in (*features.KEYS, "y")]
    n_s1 = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).height
    folds = np.random.default_rng(config.SEED + 1).integers(0, 2, n_s1).astype(np.int8)  # as train_oof.py
    fold = folds[X["i1"].to_numpy()]
    log(f"pruned training pairs with features: {X.height} ({len(names)} features)")

    oof = np.zeros(X.height, np.float32)
    boosters = []
    for f in (0, 1):
        idx = np.flatnonzero(fold == f)
        # 5% of the fold's entities for early stopping
        ent = X["i1"].to_numpy()[idx]
        val = np.isin(ent, np.random.default_rng(config.SEED + f).choice(np.unique(ent), size=len(np.unique(ent)) // 20,
                                                                           replace=False))
        M = X[idx].select(names).cast(pl.Float32).to_numpy()
        y = X["y"].to_numpy()[idx]
        dtr = lgb.Dataset(M[~val], label=y[~val], feature_name=names)
        dva = lgb.Dataset(M[val], label=y[val], reference=dtr)
        booster = lgb.train(PARAMS, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        del M, dtr, dva
        other = np.flatnonzero(fold != f)
        oof[other] = booster.predict(X[other].select(names).cast(pl.Float32).to_numpy())
        booster.save_model(str(config.artifact(f"model_pruned{tag}_{f}" if tag else f"model_pruned{f}", "txt")))
        boosters.append(booster)
        log(f"fold {f}: {booster.best_iteration} rounds; scored the other fold")
    X.select(*features.KEYS, p=pl.Series(oof, dtype=pl.Float32)).write_parquet(config.artifact(f"oof_pruned{tag}", "parquet"))
    del X

    s1, so = features.load_records("test")
    cands = metablock.prune(features.context(blocking.load_cands("test")), s1, so)
    ext = pl.read_parquet(config.artifact(f"{extra}_test", "parquet")) if extra else None
    out = []
    for T in features.build(cands, s1, so):
        if ext is not None:
            T = T.join(ext, on=features.KEYS, how="left")
        M = T.select(names).cast(pl.Float32).to_numpy()
        out.append(T.select(*features.KEYS, p=pl.Series(np.mean([b.predict(M) for b in boosters], axis=0),
                                                        dtype=pl.Float32)))
    test = pl.concat(out)
    test.write_parquet(config.artifact(f"scored_test_pruned{tag}", "parquet"))
    log(f"test pairs scored: {test.height}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default=None, help="extra feature artifact prefix, e.g. feats_v2")
    ap.add_argument("--tag", default="", help="suffix for the output artifacts, e.g. _v2")
    a = ap.parse_args()
    main(extra=a.extra, tag=a.tag)
