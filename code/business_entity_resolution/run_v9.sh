#!/usr/bin/env bash
# v9, Phase 1 of the renovation plan: context counts + refined look-alike features.
# One heavy stage at a time (31 GB laptop). Nothing here touches output/ or the v6/v7 artifacts.
# Baseline: oof_stack_v6 (stack.py --save-oof --suffix _v6). Logs go to $ER_WORK_DIR/v9_*.log.
set -euo pipefail
cd "$(dirname "$0")/src"
export ER_WORK_DIR="${ER_WORK_DIR:-c:/VSC/work/amazon-ml-challenge/work_v4}"
export ER_OUT_DIR="${ER_OUT_DIR:-c:/VSC/work/amazon-ml-challenge/output_eval}"
export PYTHONIOENCODING=utf-8
W="$ER_WORK_DIR"
run() { local name=$1; shift; echo ">>> $name $(date +%H:%M)"; "$@" > "$W/v9_$name.log" 2>&1; tail -n 3 "$W/v9_$name.log"; }

run featv2   python featv2.py                     # look-alike features (unit/suffix/digit/branch-word)
run featctx  python featctx.py                    # context counts + train/test bucket-share shift check
run pruned   python train_pruned.py --extra feats_v2,feats_ctx --tag _v9
run stack    python stack.py --second oof_pruned_v9 --primary second --strict-unseen \
                 --fixed-thr "$W/thr_v6b.json" --save-oof --suffix _v9 --dry-run
run evaluate python evaluate.py oof_stack_v9 --vs oof_stack_v6 --honest   # both tuned on the same grid
run loco_base python loco.py --sample-s1 200000 --stack                 # unseen-country cost, baseline features
run loco_v9  python loco.py --sample-s1 200000 --stack --extra feats_v2,feats_ctx
echo ">>> done $(date +%H:%M)"
