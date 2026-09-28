"""Supervised meta-blocking: shrink every entity's candidate list before matching.

Blocking keeps ~28 candidates per Source-1 entity. A light LightGBM ranker keeps
only the promising ones. Its inputs are:
  * blocking-derived signals: the pre-score, the number of shared keys, which key
    kinds matched, and the rank / relative score of the pair within the entity's
    list and among rival entities competing for the same record;
  * four cheap similarities: name and address token-set ratios, same main house
    number, and the number of shared address numbers.
A candidate is kept when the ranker's probability is >= TAU (the best candidate of
every entity is always kept), so each entity keeps only as many as it needs --
~4.7 on average. Measured on 100k held-out training entities: candidates per
entity 28.6 -> 4.7 and pair recall 0.9577 -> 0.9566, with no loss of F0.5
(0.9712 -> 0.9712). The matching model scores only the kept pairs, and they are
exactly what candidate_pairs.tsv lists.

Context signals are computed on the full blocking output before pruning, so a
kept pair has the same features it would have without pruning.

    python metablock.py        # train the ranker; tune final thresholds on pruned OOF
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import blocking
import config
import features
from decode import decode
from metric import f05

TAU = 0.01
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
              num_threads=config.N_WORKERS + 2, verbose=-1, seed=config.SEED)
_FLAGS = {f"k_{k}": b for k, b in features.KIND_BITS.items() if k != "em"}
CHEAP = ["c_tset", "a_tset", "hmain_eq", "n_inter"]
COLS = features.CTX + features.W_COLS + list(_FLAGS) + CHEAP  # W_COLS only when blocking produced them


def model_path():
    return config.WORK_DIR / "model_meta.txt"


def _keep(frame, score, tau):
    """Rows with score >= tau, plus each entity's best-scored row."""
    frame = frame.with_columns(meta=pl.Series(score, dtype=pl.Float32))
    best = pl.col("meta").rank("ordinal", descending=True).over("i1") == 1
    return frame.filter((pl.col("meta") >= tau) | best).drop("meta")


def _cheap_features(ch):
    """Same definitions as the corresponding features.pair_features columns."""
    L = pl.col
    base = ch.select(
        hmain_eq=((L("hmain_1") == L("hmain_2")) & (L("hmain_1") != "")).cast(pl.Int8),
        n_inter=L("nt_1").list.set_intersection(L("nt_2")).list.len())
    sims = pl.DataFrame({
        "c_tset": cpdist(ch["core_1"].to_list(), ch["core_2"].to_list(), scorer=fuzz.token_set_ratio,
                         workers=-1, dtype=np.float32),
        "a_tset": cpdist(ch["adf_1"].to_list(), ch["adf_2"].to_list(), scorer=fuzz.token_set_ratio,
                         workers=-1, dtype=np.float32)})
    return pl.concat([base, sims], how="horizontal")


def prune(cands, s1, so, tau=TAU, chunk=3_000_000):
    """Meta-blocking for a candidate frame that already carries context columns."""
    ranker = lgb.Booster(model_file=str(model_path()))
    r1 = s1.select("i1", core_1="core", adf_1="adf", hmain_1="hmain", nt_1="nt")
    r2 = so.select("src", "i2", core_2="core", adf_2="adf", hmain_2="hmain", nt_2="nt")
    scores = []
    for start in range(0, cands.height, chunk):
        ch = cands.slice(start, chunk)
        joined = ch.join(r1, on="i1", how="left").join(r2, on=["src", "i2"], how="left")
        X = pl.concat([ch.select(*features.CTX, *[c for c in features.W_COLS if c in ch.columns],
                                 **{n: (pl.col("kinds") & b > 0).cast(pl.Int8) for n, b in _FLAGS.items()}),
                       _cheap_features(joined)], how="horizontal")
        scores.append(ranker.predict(X.select(ranker.feature_name()).cast(pl.Float32).to_numpy()))
    return _keep(cands, np.concatenate(scores), tau)


def train_ranker(n_fit=400_000):
    """Fit the ranker on cached training features (written by train_oof.py)."""
    rng = np.random.default_rng(config.SEED + 2)
    ids = pl.Series(rng.choice(2_200_000, n_fit, replace=False).astype(np.uint32))
    scan = pl.scan_parquet(config.WORK_DIR / "feats_train" / "*.parquet")
    cols = [c for c in COLS if c in scan.collect_schema().names()]
    X = scan.select(*features.KEYS, "y", *cols).filter(pl.col("i1").is_in(ids.implode())).collect()
    booster = lgb.train(PARAMS, lgb.Dataset(X.select(cols).cast(pl.Float32).to_numpy(),
                                            label=X["y"].to_numpy(), feature_name=cols), 300)
    booster.save_model(str(model_path()))
    return booster


def _kept_train(tau):
    """Meta-blocking applied to every cached training pair (features already computed)."""
    ranker = lgb.Booster(model_file=str(model_path()))
    cols = ranker.feature_name()
    X = pl.scan_parquet(config.WORK_DIR / "feats_train" / "*.parquet").select(*features.KEYS, *cols).collect()
    score = np.concatenate([ranker.predict(X.slice(s, 5_000_000).select(cols).cast(pl.Float32).to_numpy())
                            for s in range(0, X.height, 5_000_000)])
    return _keep(X.select(*features.KEYS), score, tau)


def main(tau=TAU):
    t0 = time.time()
    train_ranker()
    print(f"ranker trained ({time.time()-t0:.0f}s)")

    kept = _kept_train(tau)
    truth = blocking._truth_pairs("train", None)
    n_s1 = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).height
    all_pairs = pl.scan_parquet(config.WORK_DIR / "feats_train" / "*.parquet").select(pl.len()).collect().item()
    hit_all = truth.join(pl.scan_parquet(config.WORK_DIR / "feats_train" / "*.parquet")
                         .select(*features.KEYS).collect(), on=features.KEYS).height
    hit_kept = truth.join(kept, on=features.KEYS).height
    print(f"train candidates per S1: {all_pairs / n_s1:.1f} -> {kept.height / n_s1:.2f} | "
          f"pair recall {hit_all / truth.height:.4f} -> {hit_kept / truth.height:.4f} ({time.time()-t0:.0f}s)")

    oof = pl.read_parquet(config.artifact("oof_train", "parquet")).join(kept, on=features.KEYS)
    s1 = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).with_row_index("i1")
    truth_idx = {i: set() for i in range(s1.height)}
    for i1, src, i2 in truth.iter_rows():
        truth_idx[i1].add((src, i2))
    by_c = {c: g["i1"].to_list() for (c,), g in s1.group_by("country")}
    base = json.loads(config.artifact("thresholds_oof", "json").read_text())
    result = {}
    for c, ids in by_c.items():
        sub = oof.filter(pl.col("i1").is_in(pl.Series(ids, dtype=pl.UInt32).implode()))
        b, best = base[c], None
        for ts in sorted({round(min(0.97, max(0.5, b["t_single"] + d)), 2) for d in (-0.05, 0, 0.05)}):
            for tm in sorted({round(min(0.9, max(0.1, b["t_match"] + d)), 2) for d in (-0.1, 0, 0.1)}):
                if tm > ts:
                    continue
                for mm in sorted({round(b["miss_mass"] * f, 2) for f in (0.5, 1, 1.5)}):
                    pred = decode(sub, ts, tm, mm)
                    f = float(np.mean([f05(pred.get(i, ()), truth_idx[i]) for i in ids]))
                    if best is None or f > best[0]:
                        best = (f, ts, tm, mm)
        f, ts, tm, mm = best
        result[c] = dict(t_single=ts, t_match=tm, miss_mass=mm, oof_pruned_f05=round(f, 5))
        print(f"  {c}: pruned-OOF F0.5={f:.4f} with t_single={ts} t_match={tm} miss_mass={mm} ({time.time()-t0:.0f}s)")
    result["_default"] = {k: max(result[c][k] for c in by_c) for k in ("t_single", "t_match", "miss_mass")}
    result["_tau"] = tau
    config.artifact("thresholds_final", "json").write_text(json.dumps(result, indent=2))
    print(f"wrote thresholds_final ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
