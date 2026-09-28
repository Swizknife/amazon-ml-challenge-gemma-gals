#!/usr/bin/env bash
# Full pipeline on a large EC2 machine (Ubuntu 24.04, >= 128 GB RAM, ideally 256 GB / 32 vCPU).
# Usage (after unzipping cloud_bundle.zip into ~/bundle):   bash ~/bundle/cloud/run_aws.sh 2>&1 | tee ~/run.log
# Downloads the official challenge dataset, installs pinned dependencies, and runs:
#   normalize -> blocking rankers (stage 0/1) -> blocking -> matcher -> out-of-fold -> meta-blocking -> predict -> validate
# Resume after a failure from a given step:   START_AT=train bash ~/bundle/cloud/run_aws.sh 2>&1 | tee -a ~/run.log
#   (steps: normalize stage0 stage1 blocking train oof metablock predict)
# Smaller machine (128 GB):   ER_N_TRAIN=800000 ER_OOF_FIT=600000 bash ~/bundle/cloud/run_aws.sh ...
set -euo pipefail

sudo apt-get update -y -q
sudo apt-get install -y -q python3-venv python3-pip unzip zip curl tmux
python3 -m venv ~/er
source ~/er/bin/activate
pip install -q --upgrade pip
pip install -q -r ~/bundle/code/business_entity_resolution/requirements.txt

ROOT=~/amazon-ml-challenge
mkdir -p "$ROOT"
cd "$ROOT"
if [ ! -d 6ab10eb3b23ba_student_resource/student_resource/dataset ]; then
  curl -L -o data.zip https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip   # official challenge data
  unzip -q data.zip -d 6ab10eb3b23ba_student_resource
fi
rm -rf "$ROOT/code" && cp -r ~/bundle/code "$ROOT/code"

# Large-machine settings (the laptop defaults are 15 / 50 / 1 / 400k / 400k); each can be overridden
export ER_K=${ER_K:-25} ER_K0=${ER_K0:-50} ER_CAP_MULT=${ER_CAP_MULT:-2} \
       ER_N_TRAIN=${ER_N_TRAIN:-1200000} ER_OOF_FIT=${ER_OOF_FIT:-900000} PYTHONIOENCODING=utf-8
cd "$ROOT/code/business_entity_resolution/src"

START_AT=${START_AT:-normalize}
go=0
step() {  # step NAME COMMAND... : runs COMMAND once NAME == START_AT has been reached
  [ "$1" = "$START_AT" ] && go=1
  if [ "$go" = 1 ]; then echo ">>> step $1 started $(date +%H:%M)"; shift; "$@"; else echo ">>> step $1 skipped"; fi
}
step normalize python -u normalize.py
step stage0    python -u blocking.py stage0
step stage1    python -u blocking.py stage1
step blocking  python -u blocking.py train test
step train     python -u train.py
step oof       python -u train_oof.py
step metablock python -u metablock.py
step predict   python -u predict.py

cd "$ROOT/6ab10eb3b23ba_student_resource/student_resource"
python utils/validate_submission.py --matching "$ROOT/output/matching_results.tsv" \
  --candidate "$ROOT/output/candidate_pairs.tsv" --test-dir dataset/test --check-ids
cp "$ROOT/work/thresholds_final.json" "$ROOT/work/thresholds.json" "$ROOT/output/"
echo "ALL DONE: results in $ROOT/output/"
