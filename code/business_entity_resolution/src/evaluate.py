"""Evaluation harness: one report for any out-of-fold pair-probability artifact.

Two evaluations of the Jev decision model (Williams, SSRN 7495638; Deng et al., arXiv
2609.24965) argue that a decision pipeline must be scored stage by stage and by consequence,
because one pooled number hides where it fails. This applies that to entity resolution, for an
artifact with columns (i1, src, i2, p) over the candidates the decoder sees (oof_pruned,
oof_stack_v6, ...), decoded exactly as the submission is (one owner + expected-F0.5 prefix):

  decompose  1 - F0.5 by consequence; the parts sum exactly to 1 - F0.5:
               singleton_merged   a singleton received a list
               emptied_findable   an entity with a true candidate received no list
               emptied_lost       an entity whose every true match was lost before matching
               false_pairs        precision loss from wrong pairs in a non-empty list
               missed_candidates  recall loss from true candidates left out
               lost_candidates    recall loss from true matches never proposed
             (sequential counterfactual: remove the false pairs, then add the missed candidates)
  calibrate  Brier, mean gap (p - y) and recall at 0.5 by house-number state, S1 name ambiguity
             (how many S1s share the exact normalized name), empty candidate address, name equality
  ablate     F0.5, wrong and correct pairs with one decoding condition removed at a time
  ceiling    decoding with p = y on the same candidates: the best any matcher could do
  honest     thresholds tuned on one entity fold and scored on the other (train_oof.py folds)
  bootstrap  --vs OTHER: paired cluster-bootstrap 95% CI of the weighted F0.5 difference; clusters
             are connected components of the candidate graph, so entities that compete for the
             same records are resampled together

    python evaluate.py oof_stack_v7 --thr <json> [--vs oof_stack_v6 --vs-thr <json>] [--honest] [--check]
Writes <WORK_DIR>/eval_<name>.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import blocking
import config
from decode import decode, one_owner
from loco import Scorer
from metric import f05
from stack import WEIGHTS

KEYS = ["i1", "src", "i2"]
GRID = [(ts, tm, mm) for ts in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8) for tm in (0.2, 0.3, 0.4)
        for mm in (0.17, 0.34, 0.5, 0.68, 0.85)]
GIANT = 0.05  # a component holding more entities than this share falls back to (country, name) groups
L = pl.col
T0 = time.time()


def log(msg):
    print(f"[eval {time.time() - T0:6.0f}s] {msg}", flush=True)


def amb_bucket(n):
    return (pl.when(n == 1).then(pl.lit("1")).when(n == 2).then(pl.lit("2"))
            .when(n <= 5).then(pl.lit("3-5")).otherwise(pl.lit("6+")))


def context(sample_s1=None):
    """Truth, per-entity truth counts and the record fields the strata need."""
    s1 = (pl.read_parquet(config.norm_path("train", 1), columns=["country", "core", "nums", "hmain"])
          .with_row_index("i1").with_columns(n_s1_name=pl.len().over("country", "core")))
    truth = blocking._truth_pairs("train", None)
    ids = s1["i1"].to_numpy()
    if sample_s1:
        ids = np.sort(np.random.default_rng(11).choice(len(ids), sample_s1, replace=False)).astype(np.uint32)
        truth = truth.filter(L("i1").is_in(pl.Series(ids).implode()))
    rec = pl.concat([pl.read_parquet(config.norm_path("train", k), columns=["core", "ad", "nums", "hmain"])
                     .with_row_index("i2").with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    return dict(s1=s1, truth=truth, n_true=truth.group_by("i1").agg(n_true=pl.len()), rec=rec, ids=ids,
                country=s1["country"].to_numpy())


def labelled(name, ctx):
    P = pl.read_parquet(config.artifact(name, "parquet"), columns=[*KEYS, "p"]).with_columns(L("p").cast(pl.Float32))
    if len(ctx["ids"]) < ctx["s1"].height:
        P = P.filter(L("i1").is_in(pl.Series(ctx["ids"]).implode()))
    return (P.join(ctx["truth"].with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left")
            .with_columns(L("y").fill_null(0)))


def by_country(P, ctx):
    out = {}
    for c in WEIGHTS:
        ids = ctx["ids"][ctx["country"][ctx["ids"]] == c]
        out[c] = (P.filter(L("i1").is_in(pl.Series(ids).implode())), ids)
    return out


def load_thr(path):
    raw = json.loads(Path(path).read_text())
    return {c: {k: float(raw[c][k]) for k in ("t_single", "t_match", "miss_mass")} for c in WEIGHTS if c in raw}


def tune(sc):
    scores = {g: sc.f(*g) for g in GRID}
    best = max(scores, key=scores.get)
    return dict(zip(("t_single", "t_match", "miss_mass"), best)), scores[best]


def f05_expr(tp, k, n):
    return (pl.when(n == 0).then((k == 0).cast(pl.Float64)).when(k == 0).then(0.0)
            .otherwise(1.25 * tp / (0.25 * n + k)))


def outcomes(sc, P, ids, n_true, t):
    """Per entity: truth count, true candidates n_ct, chosen list size k, true positives tp, F0.5."""
    ch = sc.choose(t["t_single"], t["t_match"], t["miss_mass"]).select("i1", "k", tp="cy")
    n_ct = P.group_by("i1").agg(n_ct=L("y").cast(pl.Float64).sum())
    E = (pl.DataFrame({"i1": np.asarray(ids, dtype=np.uint32)}).join(n_true, on="i1", how="left")
         .join(n_ct, on="i1", how="left").join(ch, on="i1", how="left")
         .with_columns(L("n_true").fill_null(0).cast(pl.Float64), L("n_ct").fill_null(0.0),
                       L("k").fill_null(0.0), L("tp").fill_null(0.0)))
    return E.with_columns(F=f05_expr(L("tp"), L("k"), L("n_true")))


def decompose(E):
    n = E.height
    part = (L("n_true") > 0) & (L("k") > 0)
    fa = pl.when(L("tp") > 0).then(1.25 * L("tp") / (0.25 * L("n_true") + L("tp"))).otherwise(0.0)
    fb = pl.when(L("n_ct") > 0).then(1.25 * L("n_ct") / (0.25 * L("n_true") + L("n_ct"))).otherwise(0.0)

    def mean(e):
        return float(E.select(e.cast(pl.Float64).sum()).item()) / n

    out = dict(F05=mean(L("F")),
               singleton_merged=mean((L("n_true") == 0) & (L("k") > 0)),
               emptied_findable=mean((L("n_true") > 0) & (L("k") == 0) & (L("n_ct") > 0)),
               emptied_lost=mean((L("n_true") > 0) & (L("k") == 0) & (L("n_ct") == 0)),
               false_pairs=mean(pl.when(part).then(fa - L("F")).otherwise(0.0)),
               missed_candidates=mean(pl.when(part).then(fb - fa).otherwise(0.0)),
               lost_candidates=mean(pl.when(part).then(1.0 - fb).otherwise(0.0)))
    out["candidates"] = out["emptied_lost"] + out["lost_candidates"]
    out["model_recall"] = out["emptied_findable"] + out["missed_candidates"]
    out["false_merges"] = out["singleton_merged"] + out["false_pairs"]
    out["sum_check"] = 1 - out["F05"] - out["candidates"] - out["model_recall"] - out["false_merges"]
    out.update(entities=n, singletons=int((E["n_true"] == 0).sum()),
               singletons_merged=int(((E["n_true"] == 0) & (E["k"] > 0)).sum()),
               wrong_pairs=int((E["k"] - E["tp"]).sum()), correct_pairs=int(E["tp"].sum()),
               missed_candidate_pairs=int((E["n_ct"] - E["tp"]).sum()),
               lost_pairs=int((E["n_true"] - E["n_ct"]).sum()))
    return out


def calibrate(P, ctx):
    s1 = ctx["s1"].select("i1", "country", core1="core", has1=L("nums") != "", h1="hmain",
                          ambiguity=amb_bucket(L("n_s1_name")))
    rec = ctx["rec"].select("src", "i2", core2="core", has2=L("nums") != "", h2="hmain",
                            addr_empty=(L("ad") == "") & (L("nums") == ""))
    X = (P.join(s1, on="i1").join(rec, on=["src", "i2"])
         .with_columns(house=pl.when((L("h1") == L("h2")) & (L("h1") != "")).then(pl.lit("same"))
                       .when(L("has1") & L("has2")).then(pl.lit("differ"))
                       .when(L("has1") | L("has2")).then(pl.lit("one missing")).otherwise(pl.lit("both missing")),
                       name_eq=L("core1") == L("core2"))
         .select("country", "house", "ambiguity", "addr_empty", "name_eq", "p", "y"))
    missed_all = int(((X["y"] == 1) & (X["p"] < 0.5)).sum())

    def table(frame, by):
        return (frame.group_by(by).agg(pairs=pl.len(), true_rate=L("y").cast(pl.Float64).mean(),
                                       mean_p=L("p").mean(), gap=(L("p") - L("y")).mean(),
                                       brier=((L("p") - L("y")) ** 2).mean(),
                                       recall05=((L("y") == 1) & (L("p") >= 0.5)).sum() / L("y").cast(pl.Float64).sum(),
                                       missed=((L("y") == 1) & (L("p") < 0.5)).sum())
                .with_columns(share_missed=L("missed") / missed_all).sort(by))

    tabs = {"house": table(X, ["country", "house"]), "ambiguity": table(X, ["country", "ambiguity"]),
            "addr_empty": table(X, ["country", "addr_empty"]),
            # the cell where the model was blind to name counts (2 Oct check)
            "empty_addr_same_name": table(X.filter(L("addr_empty") & L("name_eq")), ["country", "ambiguity"])}
    return {k: v.to_dicts() for k, v in tabs.items()}, tabs


def ablate(P, ids, n_true, t):
    variants = {"full": (t, True),
                "no singleton gate": (dict(t, t_single=0.0), True),
                "no minimum p": (dict(t, t_match=0.0), True),
                "no missed-match prior": (dict(t, miss_mass=0.0), True),
                "no one-owner": (t, False)}
    scorers = {True: Scorer(P.select(*KEYS, "y", "p"), ids, n_true, owner=True)}
    out = {}
    for name, (tv, owner) in variants.items():
        if owner not in scorers:
            scorers[owner] = Scorer(P.select(*KEYS, "y", "p"), ids, n_true, owner=owner)
        d = decompose(outcomes(scorers[owner], P, ids, n_true, tv))
        out[name] = {k: d[k] for k in ("F05", "wrong_pairs", "correct_pairs", "singletons_merged", "false_merges",
                                       "model_recall")}
    return out


def ceiling(P, ids, n_true):
    """Reference-oracle condition: p = y through the same decoder."""
    Q = P.with_columns(p=L("y").cast(pl.Float32))
    sc = Scorer(Q.select(*KEYS, "y", "p"), ids, n_true)
    E = outcomes(sc, Q, ids, n_true, dict(t_single=0.5, t_match=0.5, miss_mass=0.0))
    direct = E.select(pl.when(L("n_true") == 0).then(1.0).when(L("n_ct") == 0).then(0.0)
                      .otherwise(1.25 * L("n_ct") / (0.25 * L("n_true") + L("n_ct"))).mean()).item()
    f = float(E["F"].mean())
    assert abs(f - direct) < 1e-9, f"decoder disagrees with the oracle ceiling: {f} vs {direct}"
    return f


def honest(P, ids, n_true, ctx):
    """Tune on one fold, score on the other; competition for records stays country-wide."""
    folds = np.random.default_rng(config.SEED + 1).integers(0, 2, ctx["s1"].height).astype(np.int8)  # as train_oof.py
    owned = one_owner(P.select(*KEYS, "y", "p"))
    sc = {}
    for f in (0, 1):
        fid = ids[folds[ids] == f]
        sc[f] = (Scorer(owned.filter(L("i1").is_in(pl.Series(fid).implode())), fid, n_true, owner=False), len(fid))
    res = {}
    for f in (0, 1):
        t, fit = tune(sc[f][0])
        res[f] = dict(thr=t, tuned_on_own_fold=fit, scored_on_other=sc[1 - f][0].f(*t.values()))
    n0, n1 = sc[0][1], sc[1][1]
    # thresholds from fold f score fold 1 - f, so weight each by the size of the fold it scores
    return dict(folds=res, honest=(res[0]["scored_on_other"] * n1 + res[1]["scored_on_other"] * n0) / (n0 + n1))


def clusters(P, ctx):
    """Connected components of the candidate graph (S1 entity <-> record), per S1 entity."""
    n1 = ctx["s1"].height
    rec = (P.select("src", "i2").unique().with_row_index("r"))
    edges = P.select("i1", "src", "i2").join(rec, on=["src", "i2"])
    i1 = edges["i1"].to_numpy().astype(np.int64)
    r = edges["r"].to_numpy().astype(np.int64) + n1
    n = n1 + rec.height
    _, lab = connected_components(coo_matrix((np.ones(len(i1), np.int8), (i1, r)), shape=(n, n)), directed=False)
    comp = lab[:n1]
    ids = ctx["ids"]
    share = np.bincount(comp[ids]).max() / len(ids)
    if share > GIANT:
        log(f"largest component holds {share:.1%} of entities: falling back to (country, name) groups")
        comp = (ctx["s1"].select(g=pl.concat_str([L("country"), L("core")], separator="|").rank("dense"))["g"]
                .to_numpy().astype(np.int64))
        return comp, dict(kind="country_name", largest_share=float(share))
    return comp, dict(kind="components", largest_share=float(share))


def bootstrap(EA, EB, comp, draws=1000, seed=2026):
    """Weighted-F0.5 difference A - B with a Poisson cluster bootstrap (clusters never span countries)."""
    rng = np.random.default_rng(seed)
    est = np.zeros(draws)
    point = 0.0
    per = {}
    wsum = sum(WEIGHTS[c] for c in EA)
    for c in EA:
        a = EA[c].select("i1", "F")
        b = EB[c].select("i1", F_b="F")
        j = a.join(b, on="i1")
        d = (j["F"] - j["F_b"]).to_numpy()
        _, inv = np.unique(comp[j["i1"].to_numpy()], return_inverse=True)
        S, N = np.bincount(inv, weights=d), np.bincount(inv).astype(np.float64)
        e = np.empty(draws)
        for k in range(draws):
            w = rng.poisson(1.0, len(S))
            e[k] = (w @ S) / (w @ N)
        per[c] = dict(delta=float(d.mean()), ci95=[float(x) for x in np.percentile(e, [2.5, 97.5])])
        point += WEIGHTS[c] / wsum * d.mean()
        est += WEIGHTS[c] / wsum * e
    return dict(weighted_delta=float(point), weighted_ci95=[float(x) for x in np.percentile(est, [2.5, 97.5])],
                draws=draws, per_country=per)


def check(P, ids, n_true, t, n=40_000):
    """The vectorized decoder must agree with decode.decode + metric.f05."""
    sub = np.sort(np.random.default_rng(3).choice(ids, min(n, len(ids)), replace=False))
    Ps = P.filter(L("i1").is_in(pl.Series(sub).implode()))
    truth = {}
    for i1, src, i2 in Ps.filter(L("y") == 1).select(KEYS).iter_rows():
        truth.setdefault(i1, set()).add((src, i2))
    nt = n_true.filter(L("i1").is_in(pl.Series(sub).implode()))
    # truth outside the candidates still counts in the denominator
    lost = {r["i1"]: r["n_true"] - len(truth.get(r["i1"], ())) for r in nt.iter_rows(named=True)}
    for i1, k in lost.items():
        for j in range(k):
            truth.setdefault(i1, set()).add((-1, j))
    pred = decode(Ps.select(*KEYS, "p"), t["t_single"], t["t_match"], t["miss_mass"])
    ref = float(np.mean([f05(pred.get(int(i), ()), truth.get(int(i), ())) for i in sub]))
    fast = decompose(outcomes(Scorer(Ps.select(*KEYS, "y", "p"), sub, nt), Ps, sub, nt, t))["F05"]
    log(f"check on {len(sub)} entities: decode+f05 {ref:.8f} vs harness {fast:.8f}")
    assert abs(ref - fast) < 1e-6, "harness disagrees with decode.decode + metric.f05"


def evaluate(name, thr, ctx, do_honest=False, do_check=False):
    P = labelled(name, ctx)
    log(f"{name}: {P.height} pairs")
    rep, E = dict(artifact=name, countries={}), {}
    for c, (Pc, ids) in by_country(P, ctx).items():
        sc = Scorer(Pc.select(*KEYS, "y", "p"), ids, ctx["n_true"])
        t, tuned = (thr[c], None) if thr and c in thr else tune(sc)
        if do_check and c == "US":
            check(Pc, ids, ctx["n_true"], t)
        E[c] = outcomes(sc, Pc, ids, ctx["n_true"], t)
        r = dict(thresholds=t, decompose=decompose(E[c]), ablate=ablate(Pc, ids, ctx["n_true"], t),
                 ceiling=ceiling(Pc, ids, ctx["n_true"]))
        if tuned is not None:
            r["tuned_in_sample"] = tuned
        if do_honest:
            r["honest"] = honest(Pc, ids, ctx["n_true"], ctx)
        rep["countries"][c] = r
        d = r["decompose"]
        log(f"{c}: F0.5 {d['F05']:.5f} | lost to candidates {d['candidates']:.5f}, model recall "
            f"{d['model_recall']:.5f}, false merges {d['false_merges']:.5f} (check {d['sum_check']:+.1e}) | "
            f"ceiling {r['ceiling']:.5f}" + (f" | honest {r['honest']['honest']:.5f}" if do_honest else ""))
    rep["weighted_F05"] = sum(WEIGHTS[c] * rep["countries"][c]["decompose"]["F05"] for c in E) / sum(WEIGHTS[c] for c in E)
    rep["calibration"], tabs = calibrate(P, ctx)
    return rep, E, P, tabs


def show(rep, tabs):
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_width_chars(220)
    for c, r in rep["countries"].items():
        print(f"\n== {c}: ablation (F0.5, wrong pairs, correct pairs, singletons merged)")
        for v, a in r["ablate"].items():
            print(f"  {v:22s} {a['F05']:.5f}  wrong {a['wrong_pairs']:>7}  correct {a['correct_pairs']:>9}  "
                  f"singletons merged {a['singletons_merged']:>6}")
    for k, t in tabs.items():
        print(f"\n== calibration by {k}")
        print(t.with_columns(pl.col(pl.Float64, pl.Float32).round(4)))


def main(name, thr=None, vs=None, vs_thr=None, do_honest=False, do_check=False, sample_s1=None, draws=1000):
    ctx = context(sample_s1)
    log("context loaded")
    rep, E, P, tabs = evaluate(name, load_thr(thr) if thr else None, ctx, do_honest, do_check)
    show(rep, tabs)
    if vs:
        rep_b, E_b, _, _ = evaluate(vs, load_thr(vs_thr or thr) if (vs_thr or thr) else None, ctx)
        comp, how = clusters(P, ctx)
        b = bootstrap(E, E_b, comp, draws)
        rep["vs"] = dict(artifact=vs, weighted_F05=rep_b["weighted_F05"], clusters=how, **b)
        log(f"{name} - {vs}: weighted dF0.5 {b['weighted_delta']:+.5f}, 95% CI "
            f"[{b['weighted_ci95'][0]:+.5f}, {b['weighted_ci95'][1]:+.5f}] ({how['kind']})")
    out = config.WORK_DIR / f"eval_{name}{'_sample' if sample_s1 else ''}.json"
    out.write_text(json.dumps(rep, indent=2, default=float))
    log(f"weighted F0.5 {rep['weighted_F05']:.5f}; wrote {out}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="artifact with (i1, src, i2, p), e.g. oof_stack_v7")
    ap.add_argument("--thr", default=None, help="thresholds json (default: tuned in sample on a grid)")
    ap.add_argument("--vs", default=None, help="second artifact for the paired bootstrap")
    ap.add_argument("--vs-thr", default=None)
    ap.add_argument("--honest", action="store_true", help="also tune on one fold and score the other")
    ap.add_argument("--check", action="store_true", help="verify against decode.decode + metric.f05")
    ap.add_argument("--sample-s1", type=int, default=None)
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    main(a.name, a.thr, a.vs, a.vs_thr, a.honest, a.check, a.sample_s1, a.draws)
