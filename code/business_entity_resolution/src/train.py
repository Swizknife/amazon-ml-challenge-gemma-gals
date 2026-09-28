"""Train the LightGBM pair model and tune decision thresholds on a holdout.

Train and holdout are disjoint sets of Source-1 entities. The candidates of both
come from blocking run over the full training universe, so the holdout sees the
same distractors a test entity would.
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
from data_io import read_ground_truth
from decode import decode
from metric import f05

if config.USE_EMB:
    import embed

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=config.N_WORKERS + 2, verbose=-1, seed=config.SEED)
GRID = dict(t_single=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
            t_match=[0.3, 0.4, 0.5, 0.6, 0.7],
            miss_mass=[0.3, 0.6, 1.0, 1.5])


def truth_index(split, s1, so):
    """Ground truth as {i1: {(src, i2)}} plus a labelled pair frame."""
    truth = read_ground_truth(config.DATA_DIR / split / f"{split}_ground_truth.tsv")
    gt = pl.DataFrame({"s1": [s for s, v in truth.items() for _ in v],
                       "id": [x for v in truth.values() for x in v]}, schema={"s1": pl.String, "id": pl.String})
    gt = (gt.join(s1.select(pl.col("id").alias("s1"), "i1"), on="s1")
          .join(so.select("id", "src", "i2"), on="id").select("i1", "src", "i2"))
    idx = {i: set() for i in s1.filter(pl.col("id").is_in(list(truth)))["i1"].to_list()}
    for i1, src, i2 in gt.iter_rows():
        idx[i1].add((src, i2))
    return idx, gt.with_columns(y=pl.lit(1, pl.Int8))


def featurize(cands, s1, so, i1s, lab, emb=None):
    sub = cands.filter(pl.col("i1").is_in(i1s))
    X = pl.concat(list(features.build(sub, s1, so, emb=emb)))
    return X.join(lab, on=features.KEYS, how="left").with_columns(pl.col("y").fill_null(0))


def score(pred, truth, i1s):
    return float(np.mean([f05(pred.get(i, ()), truth[i]) for i in i1s])) if len(i1s) else float("nan")


def tune(hold, truth, countries, grids=None):
    """Best (t_single, t_match, miss_mass) per country on holdout predictions.
    `grids` optionally maps a country to its own grid (default: GRID)."""
    best = {}
    for country, i1s in countries.items():
        sub = hold.filter(pl.col("i1").is_in(i1s))
        results = []
        g = (grids or {}).get(country, GRID)
        for ts, tm, mm in itertools.product(g["t_single"], g["t_match"], g["miss_mass"]):
            if tm > ts:
                continue
            pred = decode(sub, ts, tm, mm)
            results.append((score(pred, truth, i1s), ts, tm, mm))
        results.sort(reverse=True)
        f, ts, tm, mm = results[0]
        best[country] = dict(t_single=ts, t_match=tm, miss_mass=mm, holdout_f05=round(f, 5))
        print(f"  {country}: F0.5={f:.4f} with t_single={ts} t_match={tm} miss_mass={mm}")
    return best


def main(n_train=config.N_TRAIN, n_hold=150_000):
    t0 = time.time()
    s1, so = features.load_records("train")
    cands = features.context(blocking.load_cands("train"))
    emb = embed.load_vectors("train") if config.USE_EMB else None
    truth, lab = truth_index("train", s1, so)
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(s1.height)
    tr_i1, ho_i1 = perm[:n_train].tolist(), perm[n_train:n_train + n_hold].tolist()
    print(f"loaded ({time.time()-t0:.0f}s)")

    Xtr = featurize(cands, s1, so, tr_i1, lab, emb)
    Xho = featurize(cands, s1, so, ho_i1, lab, emb)
    names = features.feature_names(Xtr)
    print(f"features: {len(names)} cols, train {Xtr.height} pairs (pos {Xtr['y'].mean():.3f}), "
          f"holdout {Xho.height} pairs ({time.time()-t0:.0f}s)")

    dtr = lgb.Dataset(Xtr.select(names).cast(pl.Float32).to_numpy(), label=Xtr["y"].to_numpy(), feature_name=names)
    dva = lgb.Dataset(Xho.select(names).cast(pl.Float32).to_numpy(), label=Xho["y"].to_numpy(), reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(str(config.artifact("model", "txt")))
    imp = sorted(zip(model.feature_importance("gain"), names), reverse=True)
    print("top features:", ", ".join(f"{n}" for _, n in imp[:15]))

    hold = Xho.select(*features.KEYS, p=pl.Series(model.predict(Xho.select(names).cast(pl.Float32).to_numpy())))
    hold.with_columns(y=Xho["y"]).write_parquet(config.artifact("holdout_pred", "parquet"))
    pl.DataFrame({"i1": ho_i1}).write_parquet(config.artifact("holdout_ids", "parquet"))
    country = dict(zip(s1["i1"].to_list(), s1["country"].to_list()))
    by_c = {}
    for i in ho_i1:
        by_c.setdefault(country[i], []).append(i)
    print(f"tuning thresholds ({time.time()-t0:.0f}s)")
    thr = tune(hold, truth, by_c)
    # Unseen countries (e.g. France) use the stricter of the tuned settings.
    thr["_default"] = dict(t_single=max(v["t_single"] for v in thr.values()),
                           t_match=max(v["t_match"] for v in thr.values()),
                           miss_mass=max(v["miss_mass"] for v in thr.values()))
    overall = score({**decode(hold.filter(pl.col("i1").is_in(by_c.get("US", []))), **{k: thr["US"][k] for k in ("t_single", "t_match", "miss_mass")}),
                     **decode(hold.filter(pl.col("i1").is_in(by_c.get("India", []))), **{k: thr["India"][k] for k in ("t_single", "t_match", "miss_mass")})},
                    truth, ho_i1) if {"US", "India"} <= set(by_c) else float("nan")
    thr["_holdout_overall"] = round(overall, 5)
    config.artifact("thresholds", "json").write_text(json.dumps(thr, indent=2))
    print(f"holdout macro F0.5 (all countries): {overall:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
