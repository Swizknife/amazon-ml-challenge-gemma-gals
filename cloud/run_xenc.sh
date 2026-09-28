#!/usr/bin/env bash
# Optional stage after run_aws.sh: mmBERT-small cross-encoder on borderline pairs (src/xenc.py).
# Usage:   bash ~/bundle/cloud/run_xenc.sh 2>&1 | tee ~/xenc.log   (or ~/xbundle/... from xenc_bundle.zip)
# Waits until the main run has printed ALL DONE, then runs; writes ~/amazon-ml-challenge/output_xenc/
# (a submission only if the out-of-fold F0.5 improves by >= 0.001) and ends with XENC DONE.
set -euo pipefail
ROOT=~/amazon-ml-challenge
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # the unzipped bundle

echo "waiting for the main run to finish (ALL DONE in ~/run.log) ..."
until grep -q "ALL DONE" ~/run.log 2>/dev/null; do
  if ! pgrep -f run_aws.sh >/dev/null; then
    echo "The main run is not running and has not finished: fix/resume it first (see the guide), then start this again."
    exit 1
  fi
  sleep 60
done
echo "main run finished, starting the cross-encoder stage $(date +%H:%M)"

source ~/er/bin/activate
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install -q torch
else
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -q "transformers>=4.48"
cp "$HERE/code/business_entity_resolution/src/xenc.py" "$ROOT/code/business_entity_resolution/src/"

export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_TELEMETRY=1
cd "$ROOT/code/business_entity_resolution/src"
python -u xenc.py --budget-min "${XENC_BUDGET_MIN:-120}"

if [ -f "$ROOT/output_xenc/matching_results.tsv" ]; then
  cd "$ROOT/6ab10eb3b23ba_student_resource/student_resource"
  python utils/validate_submission.py --matching "$ROOT/output_xenc/matching_results.tsv" \
    --candidate "$ROOT/output_xenc/candidate_pairs.tsv" --test-dir dataset/test --check-ids
fi
cp "$ROOT/output_xenc/xenc_report.json" ~/ 2>/dev/null || true
echo "XENC DONE"
