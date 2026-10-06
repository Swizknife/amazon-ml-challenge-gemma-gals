"""End-to-end pipeline (v10): data -> normalization -> candidates -> models -> outputs.

    python src/run_all.py            # CPU stages (part 1) and, if the cross-encoder scores exist, the final stack

Part 1 (CPU, ~9 h on a 12-thread, 31 GB laptop; every stage caches its result in work/):
  1. normalize   names/addresses, Indian-script dictionary mined from training matches
  2. rankers     blocking cascade: stage-0 (key evidence) and stage-1 (+ cheap similarities) LightGBM
                 rankers, fitted on 200k sampled training entities; ER_CA=1 adds the name x locality key
                 and ER_K0=100 widens the stage-0 shortlist (the main recall gain of v10)
  3. blocking    multi-key candidates for train and test, ranked by the cascade (top 15 per S1 and source)
  4. train       first LightGBM matcher, holdout thresholds
  5. train_oof   full-universe out-of-fold scores (+ cached training features)
  6. metablock   supervised meta-blocking ranker (28.6 -> ~5.8 candidates per S1), thresholds on pruned OOF
  7. predict     meta-blocked test candidates -> first matcher
  8. featidf     IDF-weighted overlap features for the kept training pairs
  9. pruned      second matcher trained on the meta-blocked pairs it scores (2-fold), with the IDF features
Part 2 (GPU, ~1.5 h on a Kaggle T4; src/kaggle/README.md): xenc_export.py exports the uncertain pairs
  (0.02 < p < 0.98), src/kaggle/xenc_kaggle.py fine-tunes and scores mmBERT-small (2-fold, out-of-fold),
  giving work/oof_xenc.parquet and work/scored_test_xenc.parquet.
 10. stack       collective second-stage model over both matchers' probabilities and the cross-encoder
                 features; writes output/matching_results.tsv and output/candidate_pairs.tsv.
Without the cross-encoder files the stack runs without them (the v10 pipeline minus the neural scorer).
Paths default to the challenge layout; override with ER_DATA_DIR / ER_WORK_DIR / ER_OUT_DIR (see config.py).
Heavy stages run one after another.
"""
import os

os.environ.setdefault("ER_CA", "1")      # v10 candidate generation (set before config is imported)
os.environ.setdefault("ER_K0", "100")

import shutil
import time
from pathlib import Path

import blocking
import config
import featidf
import metablock
import normalize
import predict
import stack
import train
import train_oof
import train_pruned

THRESHOLDS = Path(__file__).with_name("thresholds_v6b.json")  # fixed stack thresholds (US/India), France uses UNSEEN


def main():
    t0 = time.time()
    normalize.build()
    blocking.train_stage0()
    blocking.train_stage1()
    blocking.block_split("train")
    blocking.recall_report("train")
    blocking.block_split("test")
    train.main()
    train_oof.main()
    metablock.main()
    predict.main()
    train_pruned.kept_pairs()
    featidf.main()
    train_pruned.main(extra="feats_idf", tag="_idf")
    xenc = "oof_xenc" if config.artifact("oof_xenc", "parquet").exists() else None
    if xenc is None:
        print("no cross-encoder scores found (work/oof_xenc.parquet): stacking without them; "
              "see src/kaggle/README.md for the GPU stage")
    report = stack.main(suffix="_stack", second="oof_pruned_idf", primary="second", unseen=stack.UNSEEN,
                        fixed_thr=str(THRESHOLDS), xenc=xenc)
    if report["gain"] >= stack.MIN_GAIN:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy(f"{config.OUT_DIR}_stack/{name}", config.OUT_DIR / name)
    print(f"pipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
