"""Turn pair probabilities into per-entity match lists that maximize F0.5.

1. One-owner: a Source-2/3 record belongs to at most one Source-1 entity (true
   for every record in the training labels), so each candidate is kept only for
   the S1 that gives it the highest probability.
2. Per entity, candidates are sorted by probability and the prefix with the best
   *expected* F0.5 is kept. Expected F0.5 of a prefix of size k is approximated by
   plugging expected true positives (sum of its probabilities) and the expected
   number of true matches (sum of all probabilities + a prior for matches that
   blocking missed) into the F0.5 formula.
3. An entity whose best probability is below `t_single` gets an empty list.
"""
import numpy as np
import polars as pl


def one_owner(pairs):
    best = pl.col("p").rank("ordinal", descending=True).over("src", "i2")
    return pairs.filter(best == 1)


def _best_prefix(probs, t_single, t_match, miss_mass):
    if probs[0] < t_single:
        return 0
    expected_true = probs.sum() + miss_mass
    best_k, best_f = 0, -1.0
    tp = 0.0
    for k, p in enumerate(probs, start=1):
        if p < t_match:
            break
        tp += p
        prec, rec = tp / k, tp / expected_true
        f = 1.25 * prec * rec / (0.25 * prec + rec)
        if f > best_f:
            best_k, best_f = k, f
    return best_k


def decode(pairs, t_single=0.5, t_match=0.3, miss_mass=0.05, use_one_owner=True):
    """pairs: (i1, src, i2, p) -> {i1: [(src, i2), ...]}."""
    if use_one_owner:
        pairs = one_owner(pairs)
    pairs = pairs.sort(["i1", "p"], descending=[False, True])
    i1 = pairs["i1"].to_numpy()
    src = pairs["src"].to_numpy()
    i2 = pairs["i2"].to_numpy()
    p = pairs["p"].to_numpy()
    out = {}
    if len(i1) == 0:
        return out
    bounds = np.flatnonzero(np.diff(i1)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(i1)]):
        k = _best_prefix(p[lo:hi], t_single, t_match, miss_mass)
        if k:
            out[int(i1[lo])] = list(zip(src[lo:lo + k].tolist(), i2[lo:lo + k].tolist()))
    return out
