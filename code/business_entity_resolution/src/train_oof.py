"""Full-universe out-of-fold (OOF) scoring of the training set.

Source-1 entities are split into 2 folds; a model fitted on (a sample of) one
fold scores every candidate pair of the other fold. With a prediction for every
pair in the universe, the cross-entity steps of decoding (one-owner) run exactly
as on the test set, so the thresholds tuned here are the ones used for test.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import blocking
import config
import features
from decode import decode
from train import PARAMS, score, truth_index, tune

if config.USE_EMB:
    import embed


def main(n_fit=config.OOF_FIT, rounds=None):
    t0 = time.time()
    s1, so = features.load_records("train")
    cands = features.context(blocking.load_cands("train"))
    emb = embed.load_vectors("train") if config.USE_EMB else None
    truth, lab = truth_index("train", s1, so)
    out = config.WORK_DIR / f"feats_train{config.TAG}"
    out.mkdir(parents=True, exist_ok=True)
    if not any(out.glob("*.parquet")):
        for j, X in enumerate(features.build(cands, s1, so, emb=emb)):
            (X.join(lab, on=features.KEYS, how="left").with_columns(pl.col("y").fill_null(0))
             .write_parquet(out / f"part{j:03d}.parquet"))
        print(f"features written ({time.time()-t0:.0f}s)")
    del cands

    rng = np.random.default_rng(config.SEED + 1)
    folds = pl.DataFrame({"i1": s1["i1"], "fold": rng.integers(0, 2, s1.height).astype(np.int8)})
    X = pl.scan_parquet(out / "*.parquet").join(folds.lazy(), on="i1")
    names = [c for c in X.collect_schema().names() if c not in (*features.KEYS, "y", "fold")]
    model_rounds = rounds or lgb.Booster(model_file=str(config.artifact("model", "txt"))).current_iteration()

    preds = []
    for f in (0, 1):
        fit_ids = folds.filter(pl.col("fold") == f).sample(n=min(n_fit, folds.height // 2), seed=config.SEED)["i1"]
        fit = X.filter(pl.col("i1").is_in(fit_ids.implode())).collect()
        booster = lgb.train(PARAMS, lgb.Dataset(fit.select(names).cast(pl.Float32).to_numpy(),
                                                label=fit["y"].to_numpy(), feature_name=names),
                            num_boost_round=model_rounds)
        del fit
        # score the other fold one feature file at a time: holding all ~31M of its rows at once
        # (~10 GB) ran a 31 GB laptop out of memory once the candidate frame gained a column
        for part in sorted(out.glob("*.parquet")):
            ch = pl.read_parquet(part).join(folds, on="i1").filter(pl.col("fold") != f)
            if ch.height:
                preds.append(ch.select(*features.KEYS, p=pl.Series(
                    booster.predict(ch.select(names).cast(pl.Float32).to_numpy()), dtype=pl.Float32)))
            del ch
        print(f"fold {f} scored ({time.time()-t0:.0f}s)")
    oof = pl.concat(preds)
    oof.write_parquet(config.artifact("oof_train", "parquet"))

    by_c = {c: g["i1"].to_list() for (c,), g in s1.select("i1", "country").group_by("country")}
    # One decode over the full universe takes ~15 s, so search a small neighbourhood
    # of the holdout-tuned thresholds instead of the whole grid.
    hold_thr = json.loads(config.artifact("thresholds", "json").read_text())
    grids = {}
    for c in by_c:
        h = hold_thr[c]
        grids[c] = dict(t_single=sorted({min(0.97, max(0.5, round(h["t_single"] + d, 2))) for d in (-0.05, 0, 0.05)}),
                        t_match=sorted({min(0.9, max(0.1, round(h["t_match"] + d, 2))) for d in (-0.1, 0, 0.1)}),
                        miss_mass=sorted({round(h["miss_mass"] * f, 2) for f in (0.5, 1, 1.5)}))
    print("per-country tuning on the full universe (one-owner active):")
    thr = tune(oof, truth, by_c, grids)
    thr["_default"] = {k: max(v[k] for v in thr.values()) for k in ("t_single", "t_match", "miss_mass")}
    pred = {}
    for c, i1s in by_c.items():
        t = thr[c]
        pred.update(decode(oof.filter(pl.col("i1").is_in(i1s)), t["t_single"], t["t_match"], t["miss_mass"]))
    thr["_oof_overall"] = round(score(pred, truth, s1["i1"].to_list()), 5)
    no_owner = {}
    for c, i1s in by_c.items():
        t = thr[c]
        no_owner.update(decode(oof.filter(pl.col("i1").is_in(i1s)), t["t_single"], t["t_match"],
                               t["miss_mass"], use_one_owner=False))
    thr["_oof_without_one_owner"] = round(score(no_owner, truth, s1["i1"].to_list()), 5)
    config.artifact("thresholds_oof", "json").write_text(json.dumps(thr, indent=2))
    print(f"OOF macro F0.5: {thr['_oof_overall']:.4f} (without one-owner {thr['_oof_without_one_owner']:.4f}) "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
