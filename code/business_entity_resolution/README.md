# Business Entity Resolution (v10): reproducible pipeline

Produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data.
It uses only the provided files: no external data, APIs or lookups. Models: LightGBM (MIT) and one
fine-tuned `jhu-clsp/mmBERT-small` cross-encoder (MIT, 140M parameters; well under the 8B limit).
The cross-encoder is trained only on the provided training labels. Test text is used only for
unsupervised statistics (token frequencies, IDF); no test labels or pseudo-labels are used.

## Setup
```bash
pip install -r requirements.txt        # Python 3.13; the CPU stages need nothing else
```
Default paths (relative to the repository root): data `6ab10eb3b23ba_student_resource/student_resource/dataset/`,
intermediates `work/`, outputs `output/`. Override with `ER_DATA_DIR`, `ER_WORK_DIR`, `ER_OUT_DIR`.
Heavy stages run one after another; never two at once on a 31 GB machine.

## Run end to end
Part 1 runs on a CPU; part 2 (the cross-encoder) needs a GPU and was run on a Kaggle T4.
`run_v10.sh` and `run_v10_finish.sh` in this folder contain the exact commands used (stage timings below are
from a 12-thread, 31 GB laptop).

`python src/run_all.py` runs the CPU stages and then the final stack in one go (with the cross-encoder scores if
`work/oof_xenc.parquet` exists). The two shell scripts below are the exact commands used for the submitted files.

**Part 1: candidates, matcher, first stack** (`bash run_v10.sh`, ~9 h)
```bash
cd src
export ER_CA=1 ER_K0=100            # v10 candidate generation (see below)
python normalize.py                                    # names/addresses, Indian-script dictionary mined from train matches
python blocking.py stage0 && python blocking.py stage1 # learned rankers over the key matches
python blocking.py train test                          # candidates + recall report on train
python train.py                                        # first matcher (LightGBM)
python train_oof.py                                    # full-universe out-of-fold scores, threshold tuning
python metablock.py                                    # supervised meta-blocking (~5 candidates per S1)
python predict.py                                      # test candidates -> first matcher
python featidf.py                                      # IDF-weighted overlap features (training pairs)
python train_pruned.py --extra feats_idf --tag _idf    # second matcher on the meta-blocked pairs (+ test scoring)
python stack.py --second oof_pruned_idf --primary second --strict-unseen \
       --fixed-thr thresholds_v6b.json --save-oof --suffix _v10 --dry-run   # collective stacker (no cross-encoder yet)
```

**Part 2: cross-encoder on the uncertain pairs** (`bash run_v10_finish.sh`, ~3 h, GPU)
```bash
python xenc_export.py --tag _idf --out <dir>           # pairs with 0.02 < p < 0.98: raw name | address text
# src/kaggle/README.md: upload <dir> as a private Kaggle dataset, run src/kaggle/xenc_kaggle.py on a GPU
#   -> oof_xenc.parquet, scored_test_xenc.parquet (2-fold out-of-fold; a pair is never scored by a model
#      trained on its own entity fold; the test score is the mean of the two fold models)
python stack.py --second oof_pruned_idf --primary second --strict-unseen --xenc oof_xenc \
       --fixed-thr thresholds_v6b.json --save-oof --suffix _v10x   # writes output_v10x/{matching_results,candidate_pairs}.tsv
python evaluate.py oof_stack_v10x --vs oof_stack_v7 --honest     # paired cluster-bootstrap CI, honest thresholds
```
Validate the outputs from the `student_resource/` folder:
```bash
python utils/validate_submission.py --matching <out>/matching_results.tsv --candidate <out>/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

## What v10 changes relative to the earlier (v7) pipeline
- **Wider stage-0 shortlist (`ER_K0=100`, was 50).** Measured on train: the union of key matches holds 98.4% of
  true pairs, the stage-0 ranker's cut to 50 per source dropped that to 96.7%, and the final top-15 cut loses only
  0.05 more. Widening the stage-0 cut raises the final pair recall from 97.1% to 98.0%; only the blocking step gets
  slower, because the final kept count per business is unchanged.
- **`ca` key (`ER_CA=1`).** Exact core name x one of the record's two rarest address words, for common names.
- **IDF-weighted overlap features** (`featidf.py`) for names and addresses.
- **Cross-encoder** on the uncertain pairs, as a stacker feature (`stack.py --xenc`).
- **Evaluation harness** (`evaluate.py`): loss decomposition by consequence, calibration by stratum, decoding
  ablations, paired cluster-bootstrap confidence intervals, fold-honest thresholds. `loco.py` measures the cost of an
  unseen country (leave-one-country-out).
- Reverse retrieval (`ER_REV`) was tested and left off: +0.01-0.02 points of recall for 2.3 extra candidates per S1.

## Candidate generation in stages
1. **Blocking** (`blocking.py`): polars joins on 8 key kinds (+ `ca`): address number + rare address word; rare
   name-token pair and very rare single tokens; concatenated name; rare address-word pair; number + name token; full
   house-number structure + word; moderate name tokens for records without an address number. Token frequencies come
   from the split's own text; every key block is capped. The union (~300 matches per S1) is ranked by a learned
   cascade: stage 0 (per-kind key evidence, top `ER_K0` per source), stage 1 (+ 4 cheap string similarities, top 15).
2. **Supervised meta-blocking** (`metablock.py`): a light LightGBM ranker keeps a candidate at probability >= 0.01 (plus
   each entity's best). `candidate_pairs.tsv` is exactly the set the final model scores.

## Code map (`src/`)
| File | Purpose |
|---|---|
| `config.py` | Paths, scale knobs (`ER_K`, `ER_K0`, `ER_CA`, `ER_REV`, ...) |
| `data_io.py`, `rules.py`, `translit.py`, `normalize.py` | Safe TSV I/O, normalization lists, Indian-script dictionary mined from training matches, name/address cleaning |
| `blocking.py` | Multi-key blocking, learned cascade (`stage0`, `stage1`), recall report, diagnostics |
| `metablock.py` | Supervised meta-blocking, thresholds tuned on pruned out-of-fold scores |
| `features.py` | 66 pair features (rapidfuzz similarities, token/number/legal-form agreement, rank/rival context, key-kind flags) |
| `featidf.py`, `featv2.py`, `featctx.py` | Extra feature groups: IDF overlaps (used), look-alike and name-count features (evaluated) |
| `train.py`, `train_oof.py`, `train_pruned.py`, `train_full.py` | First matcher, out-of-fold scoring, second matcher on the pruned candidates |
| `decode.py` | One-owner assignment + expected-F0.5-optimal list selection |
| `stack.py` | Collective second stage (entity view, rival view, transitivity) + optional cross-encoder features |
| `xenc_export.py`, `kaggle/` | Export of the uncertain pairs; the Kaggle GPU job that fine-tunes and scores the cross-encoder |
| `xenc.py` | Earlier local cross-encoder implementation (CPU too slow; kept for reference) |
| `evaluate.py`, `loco.py`, `metric.py` | Evaluation harness, leave-one-country-out check, the challenge metric (`python metric.py` self-test) |
| `predict.py`, `redecode.py`, `analysis.py`, `embed.py`, `run_all.py` | Test inference, re-decoding, error analysis, an optional embedding experiment (off), the CPU pipeline runner |

## Licenses
polars (MIT), rapidfuzz (MIT), LightGBM (MIT), numpy (BSD-3), scipy (BSD-3), scikit-learn (BSD-3),
indic-transliteration (MIT), torch (BSD-3), transformers (Apache-2.0), `jhu-clsp/mmBERT-small` (MIT).
