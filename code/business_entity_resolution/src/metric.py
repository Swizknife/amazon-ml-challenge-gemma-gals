"""The challenge metric: F0.5 per Source-1 entity, macro-averaged."""


def f05(pred, true):
    pred, true = set(pred), set(true)
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred, truth, ids=None):
    """Mean F0.5 over `ids` (default: every entity in `truth`)."""
    ids = list(truth) if ids is None else list(ids)
    return sum(f05(pred.get(i, ()), truth[i]) for i in ids) / max(1, len(ids))


if __name__ == "__main__":
    # Worked example from the challenge README, plus the singleton rules.
    assert abs(f05(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"]) - 0.714) < 1e-3
    assert f05([], []) == 1.0 and f05(["S2-1"], []) == 0.0 and f05([], ["S2-1"]) == 0.0
    print("metric OK")
