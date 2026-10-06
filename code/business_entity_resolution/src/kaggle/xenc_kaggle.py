"""Kaggle GPU job for the v10 cross-encoder (pushed with `kaggle kernels push`, see kaggle/README.md).

Input: the private dataset written by src/xenc_export.py (uncertain pairs, raw text, entity folds)
  xenc_train.parquet  i1, src, i2, p, fold, y, t1, t2
  xenc_test.parquet   i1, src, i2, p, fold (-1), t1, t2
1. pilot: each candidate model fine-tunes on 80k fold-0 pairs and is scored on 20k fold-1 pairs;
   the best AUC wins (the LightGBM matcher's AUC on the same pairs is the reference)
2. full: model f fine-tunes on fold f's pairs (at most MAX_TRAIN) and scores fold 1 - f and the
   test band (one fold per GPU when two are present); test score = mean of the two folds.
   No pair is scored by a model trained on its own entity fold.
Output (/kaggle/working): oof_xenc.parquet and scored_test_xenc.parquet with (i1, src, i2, ce),
model_f0/ and model_f1/ (for scoring a new band later without retraining), report.json.
Models (MIT): jhu-clsp/mmBERT-small (140M), intfloat/multilingual-e5-small (118M).
Set XENC_ONLY=<key> to skip the pilot. Score-only mode (no training) starts automatically when a previous
run's model_f0 / model_f1 are attached as an input (kernel_sources), or with XENC_SCORE_ONLY=1.
"""
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {"mmbert": "jhu-clsp/mmBERT-small", "e5": "intfloat/multilingual-e5-small"}
LR, MAX_LEN, BS_TR, BS_INF, MAX_TRAIN, SEED = 5e-5, 64, 64, 512, 400_000, 2026
PILOT_TRAIN, PILOT_EVAL = 80_000, 20_000
WORK = "/kaggle/working"
T0 = time.time()


def log(msg):
    print(f"[xenc {time.time() - T0:6.0f}s] {msg}", flush=True)


def data_path(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def auc(y, s):
    r = np.argsort(np.argsort(s)) + 1
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(1, pos.sum() * (~pos).sum()))


def encode(tok, a, b, device):
    enc = tok(a, b, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}


@torch.inference_mode()
def predict(model, tok, a, b, device, label):
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    out = np.empty(len(a), np.float32)
    for n, s in enumerate(range(0, len(order), BS_INF)):
        idx = order[s:s + BS_INF]
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**encode(tok, [a[i] for i in idx], [b[i] for i in idx], device)).logits[:, 0]
        out[idx] = torch.sigmoid(logit.float()).cpu().numpy()
        if n % 200 == 0:
            log(f"  {label}: {min(s + BS_INF, len(order))}/{len(order)}")
    return out


def fine_tune(name, a, b, y, device):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    n_steps = max(1, len(a) // BS_TR)
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    perm = np.random.default_rng(SEED).permutation(len(a))
    yt = torch.tensor(y, dtype=torch.float32)
    for s in range(n_steps):
        idx = perm[s * BS_TR:(s + 1) * BS_TR]
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**encode(tok, [a[i] for i in idx], [b[i] for i in idx], device)).logits[:, 0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yt[idx].to(device))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if s % max(1, n_steps // 10) == 0:
            log(f"  {name} step {s}/{n_steps} loss {loss.item():.4f}")
    return model, tok


def pilot(train):
    f0, f1 = train[train.fold == 0], train[train.fold == 1]
    fit = f0.sample(min(PILOT_TRAIN, len(f0)), random_state=SEED)
    ev = f1.sample(min(PILOT_EVAL, len(f1)), random_state=SEED)
    res = {"matcher_auc": auc(ev.y.values, ev.p.values)}
    for key, name in MODELS.items():
        t = time.time()
        model, tok = fine_tune(name, fit.t1.tolist(), fit.t2.tolist(), fit.y.values, "cuda:0")
        s = predict(model, tok, ev.t1.tolist(), ev.t2.tolist(), "cuda:0", f"pilot {key}")
        res[key] = {"auc": auc(ev.y.values, s), "minutes": (time.time() - t) / 60}
        log(f"pilot {key}: AUC {res[key]['auc']:.4f} (matcher {res['matcher_auc']:.4f})")
        del model
        torch.cuda.empty_cache()
    res["chosen"] = max(MODELS, key=lambda k: res[k]["auc"])
    return res


def run_fold(f, key, device, score_only):
    train = pd.read_parquet(data_path("xenc_train.parquet"))
    test = pd.read_parquet(data_path("xenc_test.parquet"))
    path = f"{WORK}/model_f{f}"
    if score_only:
        src = glob.glob(f"/kaggle/input/**/model_f{f}", recursive=True)[0]
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForSequenceClassification.from_pretrained(src).to(device)
    else:
        fit = train[train.fold == f]
        if len(fit) > MAX_TRAIN:
            fit = fit.sample(MAX_TRAIN, random_state=SEED + f)
        log(f"fold {f}: fine-tuning {MODELS[key]} on {len(fit)} pairs ({device})")
        model, tok = fine_tune(MODELS[key], fit.t1.tolist(), fit.t2.tolist(), fit.y.values, device)
        model.save_pretrained(path)
        tok.save_pretrained(path)
    other = train[train.fold == 1 - f]
    ce = predict(model, tok, other.t1.tolist(), other.t2.tolist(), device, f"OOF fold {1 - f}")
    other[["i1", "src", "i2"]].assign(ce=ce).to_parquet(f"{WORK}/oof_f{1 - f}.parquet")
    ct = predict(model, tok, test.t1.tolist(), test.t2.tolist(), device, f"test (model {f})")
    test[["i1", "src", "i2"]].assign(ce=ct).to_parquet(f"{WORK}/test_m{f}.parquet")
    log(f"fold {f} done")


def main():
    train = pd.read_parquet(data_path("xenc_train.parquet"))
    log(f"train band {len(train)} (pos {train.y.mean():.3f}), GPUs {torch.cuda.device_count()}")
    # score-only: saved fold models are attached as an input (a previous run's output), or forced by env
    score_only = (os.environ.get("XENC_SCORE_ONLY") == "1"
                  or bool(glob.glob("/kaggle/input/**/model_f0", recursive=True)))
    report = {}
    key = os.environ.get("XENC_ONLY")
    if not key and not score_only:
        report["pilot"] = pilot(train)
        key = report["pilot"]["chosen"]
    report["model"] = MODELS.get(key, "saved fold models")
    devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() >= 2 else ["cuda:0", "cuda:0"]
    done = set()
    if devices[0] != devices[1]:  # one fold per GPU; fall back to sequential if spawning fails
        try:
            ctx = mp.get_context("spawn")
            procs = {f: ctx.Process(target=run_fold, args=(f, key, devices[f], score_only)) for f in (0, 1)}
            for p in procs.values():
                p.start()
            for f, p in procs.items():
                p.join()
                if p.exitcode == 0:
                    done.add(f)
                else:
                    log(f"fold {f} process exited with {p.exitcode}; will rerun it in-process")
        except Exception as e:  # noqa: BLE001 - any spawn problem means: run sequentially
            log(f"parallel folds unavailable ({e}); running sequentially")
    for f in (0, 1):
        if f not in done:
            run_fold(f, key, "cuda:0", score_only)
    oof = pd.concat([pd.read_parquet(f"{WORK}/oof_f{f}.parquet") for f in (0, 1)])
    oof.to_parquet(f"{WORK}/oof_xenc.parquet")
    m0, m1 = (pd.read_parquet(f"{WORK}/test_m{f}.parquet") for f in (0, 1))
    k = ["i1", "src", "i2"]
    assert (m0[k].values == m1[k].values).all(), "test rows of the two fold models are not aligned"
    m0.assign(ce=(m0.ce.values + m1.ce.values) / 2).to_parquet(f"{WORK}/scored_test_xenc.parquet")
    j = train.merge(oof, on=["i1", "src", "i2"])
    report.update(oof_pairs=len(oof), oof_auc_ce=auc(j.y.values, j.ce.values), oof_auc_matcher=auc(j.y.values, j.p.values),
                  minutes=(time.time() - T0) / 60)
    log(f"band OOF AUC: cross-encoder {report['oof_auc_ce']:.4f} vs matcher {report['oof_auc_matcher']:.4f}")
    with open(f"{WORK}/report.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
