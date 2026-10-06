#!/usr/bin/env bash
# v10 re-run (A1 + F1): name x locality key + a wider stage-0 shortlist, then IDF overlap features.
#
# A1 diagnostic (India, 2 Oct): union recall 98.4%, but the stage-0 ranker's cut to its top-50
# shortlist drops it to 96.7% (-1.75 pts) -- far more than the final top-15 cut loses (-0.05 pts).
# Reverse retrieval only recovers misses from that last, near-lossless cut (+0.01-0.02 pts for
# 2.3 extra candidates/S1: not worth it, REV stays 0). K0 controls the stage-0 cut, so it moves:
# widening it costs time only in the blocking step itself (stage-1 scores more candidates before
# still keeping the same final 15/source), not in any step after it.
#
# Works in its own folder ($ER_WORK_DIR, default work_v10; the normalized data is linked from work_v4),
# one heavy stage at a time (31 GB laptop). Logs: $ER_WORK_DIR/v10_*.log
set -euo pipefail
cd "$(dirname "$0")/src"
ROOT=c:/VSC/work/amazon-ml-challenge
export ER_WORK_DIR="${ER_WORK_DIR:-$ROOT/work_v10}" ER_OUT_DIR="${ER_OUT_DIR:-$ROOT/output_v10}"
export ER_REV="${REV:-0}" ER_CA=1 ER_CA_RANK=1 ER_K0="${K0:-100}" PYTHONIOENCODING=utf-8
W="$ER_WORK_DIR"
mkdir -p "$W"
[ -e "$W/norm" ] || cmd //c mklink //J "$(cygpath -w "$W/norm")" "$(cygpath -w "$ROOT/work_v4/norm")"
[ -e "$W/translit.json" ] || cp "$ROOT/work_v4/translit.json" "$W/"
FROM="${FROM:-}"          # resume: FROM=oof skips every stage before it (stages already completed)
run() {
  local name=$1; shift
  if [ -n "$FROM" ] && [ "$name" != "$FROM" ]; then echo ">>> skip $name"; return 0; fi
  FROM=""; echo ">>> $name $(date +%H:%M)"
  if ! "$@" > "$W/v10_$name.log" 2>&1; then echo "FAILED $name"; tail -n 20 "$W/v10_$name.log"; exit 1; fi
  grep -E "recall|per S1|F0.5|rounds|scored|saved|wrote" "$W/v10_$name.log" | tail -n 4 || true
}
run stage0    python blocking.py stage0          # rankers refitted with the ca evidence column
run stage1    python blocking.py stage1
run blocking  python blocking.py train test      # forward top-K + reverse top-r, recall report on train
run train     python train.py
run oof       python train_oof.py
run metablock python metablock.py
run predict   python predict.py
run kept      python -c "import train_pruned; train_pruned.kept_pairs()"
run featidf   python featidf.py
run pruned    python train_pruned.py --extra feats_idf --tag _idf
run stack     python stack.py --second oof_pruned_idf --primary second --strict-unseen \
                --fixed-thr "$PWD/thresholds_v6b.json" --save-oof --suffix _v10 --dry-run
cp -n "$ROOT/work_v4/oof_stack_v7.parquet" "$W/"     # the v7 baseline, compared on the same grid
run evaluate  python evaluate.py oof_stack_v10 --vs oof_stack_v7 --honest
echo ">>> done $(date +%H:%M)"
