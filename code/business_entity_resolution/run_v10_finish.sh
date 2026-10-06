#!/usr/bin/env bash
# v10, second half: cross-encoder scores for the NEW candidates, final stack, validator.
# Runs after run_v10.sh (which ends with the stack without the cross-encoder and its evaluate gate).
#   1. export the uncertain pairs of the new pipeline  2. refresh the private Kaggle dataset
#   3. score them on a Kaggle GPU with the saved fold models (no retraining)
#   4. stack with the cross-encoder features, writing the submission files  5. official validator
# One heavy stage at a time (31 GB laptop). Logs: $ER_WORK_DIR/v10f_*.log
set -euo pipefail
cd "$(dirname "$0")/src"
ROOT=c:/VSC/work/amazon-ml-challenge
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work_v10}" ER_OUT_DIR="${ER_OUT_DIR:-$ROOT/output_v10}" PYTHONIOENCODING=utf-8
W="$ER_WORK_DIR"; K="$W/kaggle_data"
TEST="$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset/test"
VALIDATOR="$ROOT/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py"
run() {
  local name=$1; shift; echo ">>> $name $(date +%H:%M)"
  if ! "$@" > "$W/v10f_$name.log" 2>&1; then echo "FAILED $name"; tail -n 20 "$W/v10f_$name.log"; exit 1; fi
  grep -E "AUC|recall|F0.5|saved|wrote|PASS|status|ready|complete" "$W/v10f_$name.log" | tail -n 4 || true
}
kstatus() { kaggle "$@" 2>&1 | tr -d '\r'; }

mkdir -p "$K" && cp -n "$ROOT/work_v4/kaggle_data/dataset-metadata.json" "$K/"
if [ -z "${SKIP_EXPORT:-}" ]; then          # SKIP_EXPORT=1: the new band is already uploaded and ready
run export python xenc_export.py --tag _idf --out "$K"
echo ">>> kaggle dataset version $(date +%H:%M)"
(cd "$K" && kaggle datasets version -p . -m "v10 band" 2>&1 | tail -n 2)
sleep 240; for i in $(seq 1 60); do [ "$(kstatus datasets status altron24/er-xenc-band)" = "ready" ] && break; sleep 20; done
fi

if [ -z "${SKIP_KAGGLE:-}" ]; then          # SKIP_KAGGLE=1: kaggle_out already holds verified scores
echo ">>> kaggle score kernel $(date +%H:%M)"
T=$(mktemp -d); cp kaggle/xenc_kaggle.py "$T/"; cp kaggle/kernel-metadata-score.json "$T/kernel-metadata.json"
(cd "$T" && kaggle kernels push -p . 2>&1 | tail -n 2)
sleep 180                                                   # a previous run's COMPLETE must not be mistaken for this one
for i in $(seq 1 120); do                                   # up to ~2 h
  s=$(kstatus kernels status altron24/er-xenc-score | grep -o 'KernelWorkerStatus\.[A-Z]*' || true)
  case "$s" in *COMPLETE*) break;; *ERROR*|*CANCEL*) echo "FAILED kaggle: $s"; exit 1;; esac; sleep 60
done
mkdir -p "$W/kaggle_out" && (cd "$W/kaggle_out" && kaggle kernels output altron24/er-xenc-score -p . --force 2>&1 | tail -n 2) || true
cp "$W/kaggle_out/oof_xenc.parquet" "$W/kaggle_out/scored_test_xenc.parquet" "$W/"
cat "$W/kaggle_out/report.json"
# the notebook must have scored exactly the exported band (a stale dataset version would not match)
python -c "import json,polars as pl,sys; n=pl.read_parquet(r'$K/xenc_train.parquet',columns=['i1']).height; m=json.load(open(r'$W/kaggle_out/report.json'))['oof_pairs']; sys.exit(0 if n==m else f'scored {m} pairs but the band has {n}: stale Kaggle dataset version')" || { echo "FAILED stale-band"; exit 1; }
fi

run stack python stack.py --second oof_pruned_idf --primary second --strict-unseen --xenc oof_xenc \
      --fixed-thr "$PWD/thresholds_v6b.json" --save-oof --suffix _v10x
run evaluate python evaluate.py oof_stack_v10x --vs oof_stack_v7 --honest
OUT="${ER_OUT_DIR}_v10x"
run validate python "$VALIDATOR" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$TEST" --check-ids
echo ">>> done $(date +%H:%M)"
