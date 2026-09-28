"""Matcher trained on meta-blocked candidates of (almost) every training entity.

train.py fits on all ~28 blocking candidates of 400k entities. After meta-blocking
the matcher only ever sees ~4.7 candidates per entity, so this variant fits on
exactly that distribution, using every training entity except the fixed 150k
holdout of train.py (features come from the cache written by train_oof.py).
It is adopted only if, on identical pruned holdout pairs, it beats the current
model by >= MIN_GAIN macro F0.5:

    python train_full.py            # writes model_full.txt + thresholds_final_full.json if it wins
    ER_TAG=_full python predict.py  # then predict with it
"""
import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import blocking
import config
import features
import metablock
from decode import decode
from metric import f05
from train import PARAMS

MIN_GAIN = 0.001


def _holdout_ids(n_train=400_000, n_hold=150_000):
    n = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).height
    perm = np.random.default_rng(config.SEED).permutation(n)  # same split as train.py
    return perm[n_train:n_train + n_hold]


def _score(pred, truth, ids):
    return float(np.mean([f05(pred.get(i, ()), truth.get(i, set())) for i in ids]))


def _decode_by_country(hold, thr, by_c):
    pred = {}
    for c, ids in by_c.items():
        t = thr[c]
        pred.update(decode(hold.filter(pl.col("i1").is_in(pl.Series(ids, dtype=pl.UInt32).implode())),
                           t["t_single"], t["t_match"], t["miss_mass"]))
    return pred


def main():
    t0 = time.time()
    kept = metablock._kept_train(metablock.TAU)
    parts = sorted((config.WORK_DIR / "feats_train").glob("*.parquet"))  # part by part: bounded memory
    X = pl.concat([pl.read_parquet(p).join(kept, on=features.KEYS) for p in parts])
    names = [c for c in X.columns if c not in (*features.KEYS, "y")]
    ho = pl.Series(_holdout_ids().astype(np.uint32))
    is_ho = pl.col("i1").is_in(ho.implode())
    tr, hv = X.filter(~is_ho), X.filter(is_ho)
    del X
    print(f"pruned pairs: train {tr.height} ({tr['i1'].n_unique()} entities), holdout {hv.height} ({time.time()-t0:.0f}s)")

    booster = lgb.train(PARAMS, lgb.Dataset(tr.select(names).cast(pl.Float32).to_numpy(), label=tr["y"].to_numpy(),
                                            feature_name=names),
                        num_boost_round=1500,
                        valid_sets=[lgb.Dataset(hv.select(names).cast(pl.Float32).to_numpy(), label=hv["y"].to_numpy())],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    del tr
    current = lgb.Booster(model_file=str(config.WORK_DIR / "model.txt"))
    Xh = hv.select(names).cast(pl.Float32).to_numpy()
    p_new = hv.select(*features.KEYS, p=pl.Series(booster.predict(Xh), dtype=pl.Float32))
    p_cur = hv.select(*features.KEYS, p=pl.Series(current.predict(Xh[:, [names.index(n) for n in current.feature_name()]]),
                                                  dtype=pl.Float32))

    truth = {}
    for i1, src, i2 in blocking._truth_pairs("train", None).filter(pl.col("i1").is_in(ho.implode())).iter_rows():
        truth.setdefault(i1, set()).add((src, i2))
    country = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).with_row_index("i1").filter(
        pl.col("i1").is_in(ho.implode()))
    by_c = {c: g["i1"].to_list() for (c,), g in country.group_by("country")}
    thr = json.loads((config.WORK_DIR / "thresholds_final.json").read_text())
    ids = [i for v in by_c.values() for i in v]
    f_cur = _score(_decode_by_country(p_cur, thr, by_c), truth, ids)
    f_new = _score(_decode_by_country(p_new, thr, by_c), truth, ids)
    print(f"holdout macro F0.5 on pruned candidates: current model {f_cur:.4f} | full-data model {f_new:.4f} "
          f"({time.time()-t0:.0f}s)")
    if f_new - f_cur < MIN_GAIN:
        print(f"gain {f_new - f_cur:+.4f} < {MIN_GAIN}: keep the current model")
        return

    # Small local re-tune of the thresholds for the new model on the holdout.
    new_thr = dict(thr)
    for c, cids in by_c.items():
        b = thr[c]
        sub = p_new.filter(pl.col("i1").is_in(pl.Series(cids, dtype=pl.UInt32).implode()))
        grid = itertools.product(sorted({round(min(0.97, max(0.5, b["t_single"] + d)), 2) for d in (-0.05, 0, 0.05)}),
                                 sorted({round(min(0.9, max(0.1, b["t_match"] + d)), 2) for d in (-0.1, 0, 0.1)}),
                                 sorted({round(b["miss_mass"] * f, 2) for f in (0.5, 1, 1.5)}))
        best = max(((_score(decode(sub, ts, tm, mm), truth, cids), ts, tm, mm) for ts, tm, mm in grid if tm <= ts))
        new_thr[c] = dict(t_single=best[1], t_match=best[2], miss_mass=best[3], holdout_f05=round(best[0], 5))
        print(f"  {c}: {best[0]:.4f} with {best[1:]}")
    new_thr["_default"] = {k: max(new_thr[c][k] for c in by_c) for k in ("t_single", "t_match", "miss_mass")}
    booster.save_model(str(config.WORK_DIR / "model_full.txt"))
    (config.WORK_DIR / "thresholds_final_full.json").write_text(json.dumps(new_thr, indent=2))
    print(f"adopted: gain {f_new - f_cur:+.4f}; wrote model_full.txt + thresholds_final_full.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
