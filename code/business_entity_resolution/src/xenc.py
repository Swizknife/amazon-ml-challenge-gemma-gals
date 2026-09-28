"""Cross-encoder re-scoring of borderline pairs (mmBERT-small, MIT, 2025).

The LightGBM matcher is confident on most pairs. Only the pairs it is unsure about
(|p - 0.5| < w) are re-read by a small multilingual transformer that sees both raw
records at once (a cross-encoder, `jhu-clsp/mmBERT-small`, 140M parameters). A small
LightGBM stacker then combines the matcher probability with the cross-encoder
score and its rank / margin within the entity and among rival entities.

Everything is out-of-fold on the same two folds as train_oof.py:
  * cross-encoder fold f is fine-tuned on borderline pairs of fold f and scores
    fold 1-f (test pairs: mean of both fold models);
  * stacker fold f is fitted on fold 1-f and scores fold f.
Thresholds are re-tuned on the pruned out-of-fold predictions exactly as in
metablock.py, and the stage writes a submission only if the weighted F0.5
(US 0.45 / India 0.55, the test mix) improves by at least MIN_GAIN.

The band width w is chosen from the measured throughput so the transformer work
fits in --budget-min minutes (most uncertain pairs first).

    python xenc.py [--budget-min 120] [--max-train 300000] [--sample-s1 N] [--device auto]
Output (when it helps): <OUT_DIR>_xenc/matching_results.tsv + candidate_pairs.tsv + xenc_report.json
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import blocking
import config
import metablock
from data_io import write_id_lists
from decode import decode
from metric import f05

MODEL = "jhu-clsp/mmBERT-small"
MAX_LEN = 64
LR = 5e-5
MIN_GAIN = 0.001
WEIGHTS = {"US": 0.45, "India": 0.55}
KEYS = ["i1", "src", "i2"]
STACK_FEATS = ["p", "ce", "ce_rank", "ce_n", "ce_gap_i1", "ce_rival", "ce_margin", "p_rank_i1", "p_max_i1", "p_rival"]
STACK_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                    feature_fraction=0.9, num_threads=config.N_WORKERS + 2, verbose=-1, seed=config.SEED)
T0 = time.time()


def log(msg):
    print(f"[xenc {time.time() - T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data
def records(split):
    """Raw 'name | address' text for S1 (i1) and S2/S3 (src, i2), indexed as everywhere else."""
    txt = pl.concat_str([pl.col("business_name"), pl.col("business_address")], separator=" | ")
    s1 = (pl.read_parquet(config.norm_path(split, 1), columns=["id", "business_name", "business_address", "country"])
          .with_row_index("i1").select("i1", "id", "country", t1=txt))
    so = pl.concat([pl.read_parquet(config.norm_path(split, k), columns=["id", "business_name", "business_address"])
                    .with_row_index("i2").select("i2", "id", t2=txt).with_columns(src=pl.lit(k, pl.Int8))
                    for k in (2, 3)])
    return s1, so


def kept_train(sample):
    """Meta-blocking applied to the cached training pairs (as metablock._kept_train, optionally sampled)."""
    ranker = lgb.Booster(model_file=str(metablock.model_path()))
    cols = ranker.feature_name()
    scan = pl.scan_parquet(config.WORK_DIR / "feats_train" / "*.parquet").select(*KEYS, *cols)
    if sample is not None:
        scan = scan.filter(pl.col("i1").is_in(sample.implode()))
    X = scan.collect()
    score = np.concatenate([ranker.predict(X.slice(s, 5_000_000).select(cols).cast(pl.Float32).to_numpy())
                            for s in range(0, X.height, 5_000_000)])
    return metablock._keep(X.select(KEYS), score, metablock.TAU)


def add_context(pairs):
    """Matcher-probability context over all pruned candidates (same for train OOF and test)."""
    top2 = pl.col("p").top_k(2).min().over("src", "i2")
    m1 = pl.col("p").max().over("src", "i2")
    n = pl.len().over("src", "i2")
    return pairs.with_columns(
        p_rank_i1=pl.col("p").rank("ordinal", descending=True).over("i1").cast(pl.Float32),
        p_max_i1=pl.col("p").max().over("i1").cast(pl.Float32),
        p_rival=pl.when(n == 1).then(-1.0).when(pl.col("p") >= m1).then(top2).otherwise(m1).cast(pl.Float32))


def stack_features(b):
    """Cross-encoder rank / gap within the entity and against rival entities (borderline rows only)."""
    top2 = pl.col("ce").top_k(2).min().over("src", "i2")
    m1 = pl.col("ce").max().over("src", "i2")
    n = pl.len().over("src", "i2")
    b = b.with_columns(
        ce_rank=pl.col("ce").rank("ordinal", descending=True).over("i1").cast(pl.Float32),
        ce_n=pl.len().over("i1").cast(pl.Float32),
        ce_gap_i1=(pl.col("ce") - pl.col("ce").max().over("i1")).cast(pl.Float32),
        ce_rival=pl.when(n == 1).then(-1.0).when(pl.col("ce") >= m1).then(top2).otherwise(m1).cast(pl.Float32))
    return b.with_columns(ce_margin=(pl.col("ce") - pl.col("ce_rival")).cast(pl.Float32))


# ----------------------------------------------------------------------------- transformer
def autocast(device, dtype):
    return torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32)


def encode(tok, a, b, device):
    enc = tok(a, b, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}


@torch.inference_mode()
def predict(model, tok, a, b, device, dtype, bs, label):
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    out = np.empty(len(a), np.float32)
    step = max(1, len(order) // 10)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        with autocast(device, dtype):
            logit = model(**encode(tok, [a[i] for i in idx], [b[i] for i in idx], device)).logits[:, 0]
        out[idx] = torch.sigmoid(logit.float()).cpu().numpy()
        if (s // bs) % max(1, step // bs) == 0:
            log(f"  {label}: {min(s + bs, len(order))}/{len(order)}")
    return out


def fine_tune(tok, a, b, y, device, dtype, bs):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_steps = max(1, len(a) // bs)
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    perm = np.random.default_rng(config.SEED).permutation(len(a))
    yt = torch.tensor(y, dtype=torch.float32)
    for s in range(n_steps):
        idx = perm[s * bs:(s + 1) * bs]
        with autocast(device, dtype):
            logit = model(**encode(tok, [a[i] for i in idx], [b[i] for i in idx], device)).logits[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yt[idx].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if s % max(1, n_steps // 10) == 0:
            log(f"  train step {s}/{n_steps} loss {loss.item():.4f}")
    return model


def benchmark(tok, a, b, device):
    """Pairs/s for inference (bf16 vs fp32, keep the faster) and for training."""
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(device)
    n = min(len(a), 256)
    rates = {}
    dtypes = [torch.bfloat16, torch.float32] if device == "cpu" or torch.cuda.is_bf16_supported() else [torch.float32]
    for dt in dtypes:
        predict(model, tok, a[:64], b[:64], device, dt, 64, "warm-up")
        t = time.time()
        predict(model, tok, a[:n], b[:n], device, dt, 128, "bench")
        rates[dt] = n / (time.time() - t)
    dtype = max(rates, key=rates.get)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    t, steps, bs = time.time(), 4, 32
    for s in range(steps):
        with autocast(device, dtype):
            logit = model(**encode(tok, a[s * bs:(s + 1) * bs], b[s * bs:(s + 1) * bs], device)).logits[:, 0]
        logit.float().mean().backward()
        opt.step()
        opt.zero_grad()
    r_tr = steps * bs / (time.time() - t)
    del model
    return rates[dtype], r_tr, dtype


# ----------------------------------------------------------------------------- evaluation
def evaluate(oof, truth_idx, by_c, thr):
    res = {}
    for c, ids in by_c.items():
        t = thr[c]
        sub = oof.filter(pl.col("i1").is_in(pl.Series(ids, dtype=pl.UInt32).implode()))
        pred = decode(sub, t["t_single"], t["t_match"], t["miss_mass"])
        res[c] = float(np.mean([f05(pred.get(i, ()), truth_idx[i]) for i in ids]))
    return res


def tune(oof, truth_idx, by_c, base):
    """Neighbourhood grid around the base thresholds (same grid as metablock.main)."""
    out = {}
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
        out[c] = dict(t_single=ts, t_match=tm, miss_mass=mm, oof_pruned_f05=round(f, 5))
        log(f"  {c}: F0.5={f:.5f} with t_single={ts} t_match={tm} miss_mass={mm}")
    out["_default"] = {k: max(out[c][k] for c in by_c) for k in ("t_single", "t_match", "miss_mass")}
    return out


def weighted(res):
    return sum(WEIGHTS[c] * res[c] for c in WEIGHTS if c in res) / sum(WEIGHTS[c] for c in WEIGHTS if c in res)


# ----------------------------------------------------------------------------- main
def main(budget_min=120, max_train=300_000, sample_s1=None, device="auto", write=True):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    log(f"device {device}, torch threads {torch.get_num_threads()}")
    s1, so = records("train")
    folds = np.random.default_rng(config.SEED + 1).integers(0, 2, s1.height).astype(np.int8)  # as train_oof.py
    sample = None
    if sample_s1:
        sample = pl.Series("i1", np.random.default_rng(7).choice(s1.height, sample_s1, replace=False).astype(np.uint32))

    # pruned out-of-fold predictions + labels
    oof = pl.scan_parquet(config.artifact("oof_train", "parquet"))
    if sample is not None:
        oof = oof.filter(pl.col("i1").is_in(sample.implode()))
    oof = oof.collect().join(kept_train(sample), on=KEYS)
    truth = blocking._truth_pairs("train", None)
    if sample is not None:
        truth = truth.filter(pl.col("i1").is_in(sample.implode()))
    fold_of = pl.DataFrame({"i1": np.arange(s1.height, dtype=np.uint32), "fold": folds})
    oof = (add_context(oof).join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=KEYS, how="left")
           .with_columns(pl.col("y").fill_null(0)).join(fold_of, on="i1"))
    ids = sample.to_list() if sample is not None else list(range(s1.height))
    truth_idx = {i: set() for i in ids}
    for i1, src, i2 in truth.iter_rows():
        truth_idx[i1].add((src, i2))
    country = s1["country"].to_numpy()
    by_c = {}
    for i in ids:
        by_c.setdefault(country[i], []).append(i)
    base_thr = json.loads(config.artifact("thresholds_final", "json").read_text())
    base = evaluate(oof, truth_idx, by_c, base_thr)
    log(f"pruned OOF: {oof.height} pairs; base F0.5 {base} weighted {weighted(base):.5f}")

    # test pairs (pruned, scored by the matcher)
    t1, t2 = records("test")
    test = pl.read_parquet(config.artifact("scored_test", "parquet"))
    if sample is not None:
        test = test.filter(pl.col("i1") < sample_s1)
    test = add_context(test)

    # band width from the measured throughput
    tok = AutoTokenizer.from_pretrained(MODEL)
    probe = oof.filter((pl.col("p") - 0.5).abs() < 0.4).head(512)
    probe = probe.join(s1.select("i1", "t1"), on="i1").join(so.select("src", "i2", "t2"), on=["src", "i2"])
    r_inf, r_tr, dtype = benchmark(tok, probe["t1"].to_list(), probe["t2"].to_list(), device)
    budget = budget_min * 60
    n_inf = max(1000, int(r_inf * 0.6 * budget))
    n_tr = int(min(max_train, r_tr * 0.35 * budget / 2))
    unc = np.sort(np.concatenate([(oof["p"] - 0.5).abs().to_numpy()] + [(test["p"] - 0.5).abs().to_numpy()] * 2))
    w = float(min(0.48, unc[n_inf - 1] if len(unc) > n_inf else 0.48))
    log(f"throughput: inference {r_inf:.0f} pairs/s, training {r_tr:.0f} pairs/s ({dtype}); "
        f"band |p-0.5|<{w:.3f}, up to {n_tr} training pairs per fold")

    def texts(frame):
        f = frame.join(s1.select("i1", "t1"), on="i1", how="left") if "t1" not in frame.columns else frame
        return f.join(so.select("src", "i2", "t2"), on=["src", "i2"], how="left")

    border = texts(oof.filter((pl.col("p") - 0.5).abs() < w))
    tb = (test.filter((pl.col("p") - 0.5).abs() < w).join(t1.select("i1", "t1"), on="i1", how="left")
          .join(t2.select("src", "i2", "t2"), on=["src", "i2"], how="left"))
    log(f"borderline pairs: train OOF {border.height} (pos {border['y'].mean():.3f}), test {tb.height}")

    bs_pred = 512 if device == "cuda" else 128
    bs_tr = 64 if device == "cuda" else 32
    ce = np.zeros(border.height, np.float32)
    ce_test = np.zeros(tb.height, np.float32)
    for f in (0, 1):
        fit = border.filter(pl.col("fold") == f)
        if fit.height > n_tr:
            fit = fit.sample(n=n_tr, seed=config.SEED + f)
        log(f"fold {f}: fine-tuning on {fit.height} pairs")
        model = fine_tune(tok, fit["t1"].to_list(), fit["t2"].to_list(), fit["y"].to_numpy(), device, dtype, bs_tr)
        other = np.flatnonzero(border["fold"].to_numpy() != f)
        sub = border[other]
        ce[other] = predict(model, tok, sub["t1"].to_list(), sub["t2"].to_list(), device, dtype, bs_pred, f"OOF fold {1 - f}")
        ce_test += predict(model, tok, tb["t1"].to_list(), tb["t2"].to_list(), device, dtype, bs_pred, "test") / 2
        del model

    # stacker, out-of-fold on the same folds
    border = stack_features(border.with_columns(ce=pl.Series(ce)))
    tb = stack_features(tb.with_columns(ce=pl.Series(ce_test)))
    p2 = np.zeros(border.height, np.float32)
    p2_test = np.zeros(tb.height, np.float32)
    for f in (0, 1):
        tr = border.filter(pl.col("fold") != f)
        booster = lgb.train(STACK_PARAMS, lgb.Dataset(tr.select(STACK_FEATS).cast(pl.Float32).to_numpy(),
                                                      label=tr["y"].to_numpy(), feature_name=STACK_FEATS), 300)
        mask = border["fold"].to_numpy() == f
        p2[mask] = booster.predict(border.filter(pl.Series(mask)).select(STACK_FEATS).cast(pl.Float32).to_numpy())
        p2_test += booster.predict(tb.select(STACK_FEATS).cast(pl.Float32).to_numpy()) / 2
    auc = lambda y, s: float(((np.argsort(np.argsort(s)) + 1)[y == 1].sum() - (y == 1).sum() * ((y == 1).sum() + 1) / 2)
                             / max(1, (y == 1).sum() * (y == 0).sum()))
    yb = border["y"].to_numpy()
    log(f"borderline AUC: matcher {auc(yb, border['p'].to_numpy()):.4f} | cross-encoder {auc(yb, ce):.4f} | "
        f"stacked {auc(yb, p2):.4f}")

    upd = border.select(*KEYS, p2=pl.Series(p2, dtype=pl.Float32))
    new_oof = (oof.join(upd, on=KEYS, how="left")
               .with_columns(p=pl.coalesce("p2", pl.col("p").cast(pl.Float32))).select(*KEYS, "p"))
    log("re-tuning thresholds on the updated pruned OOF")
    thr = tune(new_oof, truth_idx, by_c, base_thr)
    new = {c: thr[c]["oof_pruned_f05"] for c in by_c}
    gain = weighted(new) - weighted(base)
    report = dict(model=MODEL, device=device, dtype=str(dtype), band=w, n_border_train=border.height,
                  n_border_test=tb.height, base=base, new=new, weighted_base=weighted(base),
                  weighted_new=weighted(new), gain=gain, min_gain=MIN_GAIN)
    log(f"weighted F0.5 {weighted(base):.5f} -> {weighted(new):.5f} (gain {gain:+.5f}; gate {MIN_GAIN})")

    out_dir = Path(f"{config.OUT_DIR}_xenc")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "xenc_report.json").write_text(json.dumps(report, indent=2, default=str))
    if gain < MIN_GAIN or not write or sample is not None:
        log("not writing a submission (gain below gate, or sample / dry run)")
        return report

    # test submission with the stacked probabilities
    upd_t = tb.select(*KEYS, p2=pl.Series(p2_test, dtype=pl.Float32))
    scored = (test.join(upd_t, on=KEYS, how="left")
              .with_columns(p=pl.coalesce("p2", pl.col("p").cast(pl.Float32))).select(*KEYS, "p"))
    ids2 = {k: t2.filter(pl.col("src") == k).sort("i2")["id"].to_list() for k in (2, 3)}
    s1_ids = t1["id"].to_list()
    pred = {}
    for c in t1["country"].unique().to_list():
        t = thr.get(c, thr["_default"])
        pred.update(decode(scored.filter(pl.col("i1").is_in(t1.filter(pl.col("country") == c)["i1"].implode())),
                           t["t_single"], t["t_match"], t["miss_mass"]))
    matches = {s1_ids[i1]: [ids2[src][i2] for src, i2 in lst] for i1, lst in pred.items()}
    cand = scored.join(t2.select("src", "i2", "id"), on=["src", "i2"]).group_by("i1").agg(pl.col("id"))
    candidates = {s1_ids[i1]: lst for i1, lst in cand.iter_rows()}
    write_id_lists(out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, matches)
    write_id_lists(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, candidates)
    (out_dir / "thresholds_xenc.json").write_text(json.dumps(thr, indent=2))
    n_match = sum(len(v) for v in matches.values())
    log(f"wrote {out_dir}: {len(matches)} of {len(s1_ids)} S1 matched, {n_match / len(s1_ids):.2f} links per S1")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=120)
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--sample-s1", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.budget_min, a.max_train, a.sample_s1, a.device, write=not a.dry_run)
