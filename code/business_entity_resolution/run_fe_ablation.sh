#!/usr/bin/env bash
# Feature-engineering ablation (renovation plan, Phase 1): which feature group helps, measured.
# Each variant retrains the second matcher (out-of-fold only), re-runs the collective stacker and
# is compared with the v6 baseline by a paired cluster bootstrap (evaluate.py --vs).
#   _ctx  context counts (featctx.py)          _fv2  look-alike features (featv2.py)
#   _v9   both
# One heavy stage at a time (31 GB laptop). Logs: $ER_WORK_DIR/fe_*.log
set -euo pipefail
cd "$(dirname "$0")/src"
export ER_WORK_DIR="${ER_WORK_DIR:-c:/VSC/work/amazon-ml-challenge/work_v4}"
export ER_OUT_DIR="${ER_OUT_DIR:-c:/VSC/work/amazon-ml-challenge/output_eval}"
export PYTHONIOENCODING=utf-8
W="$ER_WORK_DIR"
V7_THR="c:/VSC/work/amazon-ml-challenge/submissions/final_v7/thresholds_stack.json"
run() {
  local name=$1; shift; echo ">>> $name $(date +%H:%M)"
  if ! "$@" > "$W/fe_$name.log" 2>&1; then echo "FAILED $name"; tail -n 20 "$W/fe_$name.log"; exit 1; fi
  grep -E "dF0.5|F0.5 [0-9]|weighted F0.5|fold [01]:|saved|wrote" "$W/fe_$name.log" | tail -n 6 || true
}

# re-verify: v7 vs v6 with their submission thresholds must be a statistical tie (both scored 0.968)
run verify_v7_v6 python evaluate.py oof_stack_v7 --thr "$V7_THR" --vs oof_stack_v6 --vs-thr "$W/thr_v6b.json"

run featv2  python featv2.py
run featctx python featctx.py
for v in "_ctx feats_ctx" "_fv2 feats_v2" "_v9 feats_v2,feats_ctx"; do
  set -- $v; tag=$1; extra=$2
  run "pruned$tag" python train_pruned.py --extra "$extra" --tag "$tag" --oof-only
  run "stack$tag"  python stack.py --second "oof_pruned$tag" --primary second --strict-unseen \
                     --fixed-thr "$W/thr_v6b.json" --save-oof --suffix "$tag" --dry-run
  run "eval$tag"   python evaluate.py "oof_stack$tag" --vs oof_stack_v6     # both tuned on the same grid
done
echo ">>> done $(date +%H:%M)"
