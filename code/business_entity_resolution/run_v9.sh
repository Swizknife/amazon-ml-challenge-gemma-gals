#!/usr/bin/env bash
# v9 experiment chain (REDESIGN_PLAN.md §3.1-3.2), strictly one heavy stage at a time.
# Baseline: v6 (weighted pruned OOF 0.97887). Nothing here touches output/ or the v6/v7 artifacts.
set -euo pipefail
cd "$(dirname "$0")/src"
export ER_WORK_DIR="${ER_WORK_DIR:-c:/VSC/work/amazon-ml-challenge/work_v4}"
export ER_OUT_DIR="${ER_OUT_DIR:-c:/VSC/work/amazon-ml-challenge/output_v4_v9}"
export PYTHONIOENCODING=utf-8
python featv2.py                                   # look-alike features for kept train / pruned test pairs
python loco.py --extra feats_v2_train              # does v2 shrink the unseen-country cost?
python train_pruned.py --extra feats_v2 --tag _v2  # second matcher with v2 features
python stack.py --second oof_pruned_v2 --primary second --wide-grid --strict-unseen --baseline 0.97887 --suffix _stack
