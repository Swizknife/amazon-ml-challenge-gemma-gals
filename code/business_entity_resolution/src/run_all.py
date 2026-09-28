"""End-to-end pipeline: data -> normalization -> blocking -> models -> outputs.

    python src/run_all.py

Stages (each caches its result in work/):
  1. normalize   names/addresses, Indian-script dictionary mined from training matches
  2. rankers     blocking cascade: stage-0 (key evidence) and stage-1 (+ cheap similarities)
                 LightGBM rankers, fitted on 200k sampled training entities
  3. blocking    multi-key candidates for train and test, ranked by the cascade
                 (top 15 per S1 and source) (+ train recall report)
  4. train       LightGBM matching model, holdout thresholds
  5. train_oof   full-universe out-of-fold scores (+ cached training features)
  6. metablock   supervised meta-blocking ranker, thresholds tuned on pruned OOF scores
  7. predict     meta-blocked test candidates -> matching model -> F0.5-optimal decoding
  8. pruned      second matcher trained on the meta-blocked candidates it scores (2-fold)
  9. stack       collective second-stage model over both matchers' probabilities (rival
                 entities, co-matched records, transitivity); its outputs replace output/
                 when it improves the out-of-fold F0.5 by >= 0.001
Paths default to the challenge layout; override with ER_DATA_DIR / ER_WORK_DIR /
ER_OUT_DIR (see config.py). Heavy stages run one after another (~2.5 h on a
12-thread, 31 GB laptop; ~8.5 h with the cascade).
"""
import shutil
import time

import blocking
import config
import metablock
import normalize
import predict
import stack
import train
import train_oof
import train_pruned

if __name__ == "__main__":
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
    train_pruned.main()
    report = stack.main(suffix="_stack", second="oof_pruned", primary="second", wide=True, unseen=stack.UNSEEN,
                        iterate=True)
    if report["gain"] >= stack.MIN_GAIN:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy(f"{config.OUT_DIR}_stack/{name}", config.OUT_DIR / name)
    print(f"pipeline finished in {(time.time() - t0) / 60:.1f} min")
