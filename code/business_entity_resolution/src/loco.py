"""Leave-one-country-out (LOCO): what an unseen country costs, and which decoding policy
an unseen country (France) should use.

France has no labels, so its thresholds cannot be tuned; stack.UNSEEN is a hand-picked
guess. LOCO simulates France on train: the pruned matcher (train_pruned.py pairs,
features and parameters) is fitted on one country and scores the other, which it has
never seen. On each held-out country we compare
  seen      in-country out-of-fold scores (oof_pruned), thresholds tuned on the grid
  unseen    LOCO scores decoded with the current France policy (stack.UNSEEN)
  borrowed  LOCO scores decoded with the *other* country's tuned thresholds
  policy    LOCO scores over the grid; the policy is the grid point with the best mean
            F0.5 over both held-out directions (never tuned on one country alone)
  calib     LOCO scores after an isotonic calibration fitted on the *other* direction's
            LOCO scores (cross-fitted), then the same policy search
With --stack, a collective stacker (stack.py features, p = pruned probability) is fitted
on the training country's out-of-fold scores and applied to the held-out country's LOCO
scores, so the policy is found on the same kind of probability the submission decodes.

    python loco.py [--sample-s1 N] [--stack] [--check] [--extra feats_v2,feats_ctx]
Writes <WORK_DIR>/loco_report[_<extras>].json (--extra joins <prefix>_train feature artifacts, as train_pruned.py)
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

import blocking
import config
import features
import stack
from decode import decode, one_owner
from metric import f05
from train import PARAMS
from train_pruned import kept_pairs

KEYS = features.KEYS
GRID_TS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
GRID_TM = (0.2, 0.3, 0.4, 0.5)
GRID_MM = (0.17, 0.34, 0.5)
BIAS_GRID = (0.0, -0.1, -0.2, -0.3, -0.5, -0.8)
T0 = time.time()


def log(msg):
    print(f"[loco {time.time() - T0:6.0f}s] {msg}", flush=True)


class Scorer:
    """Vectorized decode.decode + macro F0.5 for one country and one probability column.

    decode() keeps, per entity, the probability-sorted prefix maximizing expected F0.5
    = 1.25 * sum(p[:k]) / (0.25 * (sum(p) + miss_mass) + k); the realized F0.5 of a
    prefix is 1.25 * tp / (0.25 * n_true + k). Both reduce to cumulative sums over the
    one-owner, sorted pairs, so a whole threshold grid costs one sort.
    """

    def __init__(self, pairs, ids, n_true, owner=True):
        P = (one_owner(pairs) if owner else pairs).sort(["i1", "p"], descending=[False, True])
        P = P.with_columns(k=pl.int_range(1, pl.len() + 1).over("i1").cast(pl.Float64),
                           ctp=pl.col("p").cast(pl.Float64).cum_sum().over("i1"),
                           cy=pl.col("y").cast(pl.Float64).cum_sum().over("i1"),
                           psum=pl.col("p").cast(pl.Float64).sum().over("i1"),
                           pmax=pl.col("p").max().over("i1"))
        self.P = P.join(n_true, on="i1", how="left").with_columns(pl.col("n_true").fill_null(0).cast(pl.Float64))
        ent = pl.DataFrame({"i1": np.asarray(ids, dtype=np.uint32)}).join(n_true, on="i1", how="left")
        self.n_ids = ent.height
        # entities that end with an empty list score 1 if they are singletons, else 0
        self.empty_ok = int((ent["n_true"].fill_null(0) == 0).sum())

    def choose(self, ts, tm, mm):
        """The chosen prefix of every entity that receives a non-empty list: its length k and true positives cy."""
        L = pl.col
        return (self.P.filter((L("pmax") >= ts) & (L("p") >= tm))
                .with_columns(fexp=1.25 * L("ctp") / (0.25 * (L("psum") + mm) + L("k")))
                .sort(["i1", "fexp", "k"], descending=[False, True, False])
                .group_by("i1", maintain_order=True).first())

    def f(self, ts, tm, mm):
        L = pl.col
        best = self.choose(ts, tm, mm)
        # realized F0.5 of each chosen prefix; singletons that got a list score 0
        real = best.select(pl.when(L("n_true") == 0).then(0.0)
                           .otherwise(1.25 * L("cy") / (0.25 * L("n_true") + L("k"))).sum()).item()
        lost_singletons = int((best["n_true"] == 0).sum())
        return (real + self.empty_ok - lost_singletons) / self.n_ids


def grid(scorer):
    return {(ts, tm, mm): scorer.f(ts, tm, mm) for ts in GRID_TS for tm in GRID_TM for mm in GRID_MM if tm <= ts}


def fit(X, names, rows, seed, max_rounds=2000):
    ent = X["i1"].to_numpy()[rows]
    u = np.unique(ent)
    val = np.isin(ent, np.random.default_rng(seed).choice(u, size=max(1, len(u) // 20), replace=False))
    M = X[rows].select(names).cast(pl.Float32).to_numpy()
    y = X["y"].to_numpy()[rows]
    dtr = lgb.Dataset(M[~val], label=y[~val], feature_name=names)
    dva = lgb.Dataset(M[val], label=y[val], reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def check(pairs, ids, truth_idx, scorer, t):
    """The vectorized scorer must agree with decode.decode + metric.f05."""
    pred = decode(pairs.select(*KEYS, "p"), t["t_single"], t["t_match"], t["miss_mass"])
    ref = float(np.mean([f05(pred.get(i, ()), truth_idx.get(i, ())) for i in ids]))
    fast = scorer.f(t["t_single"], t["t_match"], t["miss_mass"])
    log(f"scorer check: decode+f05 {ref:.6f} vs vectorized {fast:.6f}")
    assert abs(ref - fast) < 1e-4, "vectorized scorer disagrees with decode.decode"


def stacked_loco(P, held, so):
    """Collective stacker fitted on the other country (p = in-country OOF) and applied to
    the held-out country (p = LOCO), as the submission's stacker meets France."""
    mixed = P.with_columns(p=pl.when(pl.col("country") == held).then("p_loco").otherwise("p_seen"))
    X = stack.features(mixed.select(*KEYS, "y", "country", "p"), so)
    tr = X.filter(pl.col("country") != held)
    b = lgb.train(stack.PARAMS, lgb.Dataset(tr.select(stack.FEATS).cast(pl.Float32).to_numpy(),
                                            label=tr["y"].to_numpy(), feature_name=stack.FEATS), stack.ROUNDS)
    te = X.filter(pl.col("country") == held)
    return te.select(*KEYS, p_stack=pl.Series(b.predict(te.select(stack.FEATS).cast(pl.Float32).to_numpy()),
                                              dtype=pl.Float32))


def main(sample_s1=None, do_stack=False, do_check=False, extra=None, bias_thr=None):
    s1 = pl.read_parquet(config.norm_path("train", 1), columns=["country"]).with_row_index("i1")
    country = s1["country"].to_numpy()
    sample = None
    if sample_s1:
        sample = pl.Series("i1", np.random.default_rng(11).choice(s1.height, sample_s1, replace=False).astype(np.uint32))

    kept = kept_pairs()
    if sample is not None:
        kept = kept.filter(pl.col("i1").is_in(sample.implode()))
    parts = sorted((config.WORK_DIR / "feats_train").glob("*.parquet"))
    X = pl.concat([pl.read_parquet(p).join(kept, on=KEYS) for p in parts])
    for e in (extra.split(",") if extra else []):
        X = X.join(pl.read_parquet(config.artifact(f"{e}_train", "parquet")), on=KEYS, how="left")
    names = [c for c in X.columns if c not in (*KEYS, "y")]
    c_pair = country[X["i1"].to_numpy()]
    countries = sorted(set(c_pair.tolist()))
    log(f"pairs: {X.height} ({len(names)} features); countries {countries}")

    p_loco = np.zeros(X.height, np.float32)
    rounds = {}
    for held in countries:
        tr, te = np.flatnonzero(c_pair != held), np.flatnonzero(c_pair == held)
        b = fit(X, names, tr, config.SEED + 7)
        p_loco[te] = b.predict(X[te].select(names).cast(pl.Float32).to_numpy())
        rounds[held] = b.best_iteration
        log(f"trained without {held} ({b.best_iteration} rounds); scored {len(te)} {held} pairs")
    P = X.select(*KEYS, "y").with_columns(p_loco=pl.Series(p_loco), country=pl.Series(c_pair))
    del X
    seen = pl.read_parquet(config.artifact("oof_pruned", "parquet")).select(*KEYS, p_seen=pl.col("p").cast(pl.Float32))
    P = P.join(seen, on=KEYS, how="left")
    assert P["p_seen"].null_count() == 0, "oof_pruned does not cover the kept pairs"

    truth = blocking._truth_pairs("train", None)
    if sample is not None:
        truth = truth.filter(pl.col("i1").is_in(sample.implode()))
    n_true = truth.group_by("i1").agg(n_true=pl.len())
    ids_all = sample.to_numpy() if sample is not None else np.arange(s1.height, dtype=np.uint32)
    ids_by_c = {c: ids_all[country[ids_all] == c] for c in countries}

    cols = ["p_seen", "p_loco"]
    if do_stack:
        so = stack.other_records("train")
        st = pl.concat([stacked_loco(P, held, so) for held in countries])
        P = P.join(st, on=KEYS, how="left")
        cols.append("p_stack")
        log("stacked LOCO scores built")

    # cross-fitted calibration: isotonic fitted on one held-out direction, applied to the other
    for col in cols[1:]:
        cal = np.zeros(P.height, np.float32)
        for held in countries:
            other = P.filter(pl.col("country") != held)
            iso = IsotonicRegression(out_of_bounds="clip").fit(other[col].to_numpy(), other["y"].to_numpy())
            m = (P["country"] == held).to_numpy()
            cal[m] = iso.predict(P[col].to_numpy()[m])
        P = P.with_columns(pl.Series(f"{col}_cal", cal))
        cols.append(f"{col}_cal")

    base = json.loads(config.artifact("thresholds_final", "json").read_text())
    unseen = stack.UNSEEN
    report = dict(pairs=P.height, rounds=rounds, sample_s1=sample_s1, grid=dict(ts=GRID_TS, tm=GRID_TM, mm=GRID_MM),
                  countries={})
    grids = {}
    for c in countries:
        sub = P.filter(pl.col("country") == c)
        ids = ids_by_c[c]
        rep = {}
        for col in cols:
            sc = Scorer(sub.select(*KEYS, "y", p=col), ids, n_true)
            if do_check and col == "p_loco":
                truth_idx = {}
                for i1, src, i2 in truth.filter(pl.col("i1").is_in(pl.Series(ids).implode())).iter_rows():
                    truth_idx.setdefault(i1, set()).add((src, i2))
                check(sub.select(*KEYS, p=col), ids, truth_idx, sc, unseen)
            g = grids[(c, col)] = grid(sc)
            best = max(g, key=g.get)
            other = next(o for o in countries if o != c)
            rep[col] = dict(best=dict(zip(("t_single", "t_match", "miss_mass"), best)), best_f05=g[best],
                            unseen_policy_f05=sc.f(unseen["t_single"], unseen["t_match"], unseen["miss_mass"]),
                            borrowed_f05=sc.f(*(base[other][k] for k in ("t_single", "t_match", "miss_mass"))))
            log(f"{c} {col}: best {g[best]:.5f} at {best}; UNSEEN policy {rep[col]['unseen_policy_f05']:.5f}; "
                f"{other}'s thresholds {rep[col]['borrowed_f05']:.5f}")
        report["countries"][c] = rep

    # the unseen-country policy: best mean F0.5 over both held-out directions
    report["policy"] = {}
    for col in cols[1:]:
        mean = {k: np.mean([grids[(c, col)][k] for c in countries]) for k in grids[(countries[0], col)]}
        best = max(mean, key=mean.get)
        report["policy"][col] = dict(zip(("t_single", "t_match", "miss_mass"), best),
                                     mean_f05=float(mean[best]),
                                     per_country={c: grids[(c, col)][best] for c in countries},
                                     unseen_mean_f05=float(np.mean([report["countries"][c][col]["unseen_policy_f05"]
                                                                    for c in countries])))
        log(f"policy for {col}: {best} mean F0.5 {mean[best]:.5f} "
            f"(current UNSEEN {report['policy'][col]['unseen_mean_f05']:.5f})")
    report["cost_of_unseen"] = {c: report["countries"][c]["p_seen"]["best_f05"] - report["policy"]["p_loco"]["per_country"][c]
                                for c in countries}
    log(f"cost of an unseen country (seen best - LOCO policy): {report['cost_of_unseen']}")
    if do_stack and bias_thr:
        # unseen-country logit bias: each held-out country is decoded with the *other* country's stack
        # thresholds, on LOCO-stacked probabilities shifted by b -- as France meets the submission
        thr = json.loads(Path(bias_thr).read_text())
        report["bias"] = {}
        for b in BIAS_GRID:
            per = {}
            for c in countries:
                t = thr[next(o for o in countries if o != c)]
                sub = P.filter(pl.col("country") == c)
                sub = sub.select(*KEYS, "y", p=pl.Series(stack.shift(sub["p_stack"].to_numpy(), b)))
                per[c] = Scorer(sub, ids_by_c[c], n_true).f(t["t_single"], t["t_match"], t["miss_mass"])
            report["bias"][f"{b:+.1f}"] = dict(per_country=per, mean=float(np.mean(list(per.values()))))
            log(f"unseen-country bias {b:+.1f}: {per} mean {report['bias'][f'{b:+.1f}']['mean']:.5f}")
        report["bias_best"] = float(max(report["bias"], key=lambda k: report["bias"][k]["mean"]))
        log(f"best unseen-country bias: {report['bias_best']:+.1f}")
    out = config.WORK_DIR / f"loco_report{'_' + extra.replace(',', '+') if extra else ''}{'_sample' if sample is not None else ''}.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    log(f"wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-s1", type=int, default=None)
    ap.add_argument("--stack", action="store_true", help="also fit a LOCO collective stacker")
    ap.add_argument("--check", action="store_true", help="verify the vectorized scorer against decode.decode")
    ap.add_argument("--extra", default=None, help="extra feature artifact prefixes, comma-separated, e.g. feats_v2,feats_ctx")
    ap.add_argument("--bias-thr", default=None, help="stack thresholds json; with --stack, search the unseen-country bias")
    a = ap.parse_args()
    main(a.sample_s1, a.stack, a.check, a.extra, a.bias_thr)
