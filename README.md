# Business Entity Resolution — team *gemma_gals*

**Amazon ML Challenge 2026.** Public leaderboard **F0.5 = 0.968** (final submission, 27 Sep 2026).
Team: Soumya Sharma (leader), Anjali Singh, Prisha Raj — BIT Mesra, Patna Campus.

Match 1.7M master-list businesses against ~10M noisy records from two other sources — 17 trillion
possible pairs — on one laptop, under a metric that punishes a wrong merge about twice as hard as a
miss.

| Document | What it is |
|---|---|
| **[PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md)** | **The full teaching write-up: every stage, every method, and *why*, written for someone new to ML.** Start here. |
| [submissions/final_v10/Documentation_template.md](submissions/final_v10/Documentation_template.md) | The official methodology document (v10) |
| [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md) | Per-stage commands, timings, code map |
| [HANDOFF.md](HANDOFF.md) | Complete project record: rules, EDA, submission log, AWS guide, plans |

---

## The problem in one paragraph

We get business records — name, address, country — from three independent sources (`S1`, `S2`,
`S3`), each noisy in its own way: typos, abbreviations, transliterated Indian-script names, missing
house numbers, reordered address parts. `S1` is a clean, deduplicated reference list. For **every**
`S1` business we must list every `S2`/`S3` record that is the *same real business* — zero, one, or
many. Scale rules out comparing everything with everything (1.7M × ~10M ≈ 17 trillion pairs). The
data also plants deliberate traps — "sibling" businesses sharing a building with a different unit
number — so a model that matches on name and street alone gets punished hard, because the metric
(**F0.5**) weights precision **2×** over recall: one wrong merge costs about as much as two missed
matches, and for the 5.6% of businesses with no match at all, predicting anything scores 0.

## The pipeline

```
raw .tsv files
     │
     ▼
① NORMALIZE        clean names/addresses; translate Indian-script names with a dictionary
     │             MINED FROM THE TRAINING ANSWERS (not hand-written); expand abbreviations;
     │             parse every number ("1577/15" → main 1577, sub 15)
     ▼
② BLOCKING         ~300 plausible records per S1 via 9 kinds of cheap shared keys (rare name
     │             words, house number + street word, …); a LEARNED two-stage LightGBM cascade
     │             ranks them and keeps 15 per source; SUPERVISED META-BLOCKING cuts that
     │             to ~5.8 per entity                                   ──► candidate_pairs.tsv
     ▼
③ FEATURES         73 numbers per pair: name/address similarity (rapidfuzz), house-number
     │             structure agreement, IDF-weighted overlap (a shared rare word outweighs a
     │             shared common one), and — the highest-signal group — RIVAL FEATURES: does a
     │             competing S1 fit this record better than we do?
     ▼
④ MATCHERS         LightGBM gives each pair a match probability. A SECOND matcher is trained on
     │             exactly the ~5.8-candidate distribution it will actually score (not the ~28
     │             the first one sees), so training and serving agree.
     ▼
⑤ CROSS-ENCODER    the ~13% of pairs the matchers are unsure about (0.02 < p < 0.98) are re-read
     │             by a fine-tuned multilingual transformer (mmBERT-small, 140M, MIT) that sees
     │             both records as one text. Out-of-fold AUC on those pairs: 0.955 vs 0.931.
     ▼
⑥ COLLECTIVE       a stacker re-scores every pair from both matchers' probabilities, the
     │             STACKER      cross-encoder score, and the predictions for rival entities and
     │             for the entity's other likely matches (collective classification)
     ▼
⑦ DECISION RULES   one S2/S3 record → at most ONE S1 (proven true in the data); keep the
     │             probability-sorted prefix that MAXIMIZES EXPECTED F0.5 (not "p > 0.5"); say
     │             nothing when unsure; thresholds tuned per country, France (never seen in
     │             training) uses the stricter setting
     ▼
matching_results.tsv  →  leaderboard
```

Every capitalized step exists because we **measured** a property of the data first, not because it
is the textbook default. [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md) walks through each one with
the evidence.

## Why this design

- **Blocking sets the ceiling.** A true match blocking never proposes cannot be recovered by any
  model downstream, so we invested in it as much as in the classifier: a hand-tuned multi-key
  search, *then* two learned re-ranking stages, taking candidates per entity from ~300 → 28.6 → 5.8
  while pair recall only drops 0.984 → 0.980 → 0.979.
- **F0.5 rewards being right, not being complete.** Everything after blocking is built to say "no
  match" by default: the one-owner constraint, the expected-F0.5-optimal prefix instead of a flat
  cutoff, and per-country thresholds tuned to avoid false merges on the sibling traps the dataset
  plants on purpose.
- **Collective, not just pairwise, decisions.** Two S1 businesses often compete for the same
  ambiguous record. The stacker feeds each pair the probabilities of its *rivals* — what else could
  this record be, and what else does this S1 already have — which is exactly the structure the
  sibling traps are designed to defeat.
- **The model must transfer to an unseen country.** France appears only in test. No feature is
  country-specific (no country flag anywhere), and the France thresholds were validated by a
  leave-one-country-out study ([src/loco.py](code/business_entity_resolution/src/loco.py)) rather
  than left as a guess.
- **Nothing is adopted on a hunch.** Every change had to beat a pre-declared gate (+0.001 weighted
  out-of-fold F0.5) measured by [src/evaluate.py](code/business_entity_resolution/src/evaluate.py),
  with confidence intervals from a paired cluster bootstrap. About half the ideas we liked were
  rejected — they are listed in
  [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md#things-we-tried-that-did-not-work).

## Results

**Public leaderboard** (the only real score):

| Version | What changed | LB F0.5 |
|---|---|---|
| v2 | blocking key fixes (typo-proof rarity, wider key set) | 0.963 |
| v4 | learned two-stage blocking cascade | 0.966 |
| v5 | + collective second stage | 0.967 |
| v6 | + second matcher on the pruned candidate distribution | 0.968 |
| **v7 — final submission** | + a second collective round | **0.968** |
| v10 | post-deadline: wider stage-0 shortlist, `ca` key, IDF features, cross-encoder | *not scored* |

**Internal out-of-fold** — macro F0.5 over all 2.2M training entities, every entity scored by a model
that never saw it, weighted US 0.45 / India 0.55 to match the test country mix. This is a measuring
instrument for comparing our own versions, **not** a leaderboard prediction: v7 reads 0.9790 here
and scored 0.968 publicly, because test contains France (unseen) and ~23% more distractor records.

| Out-of-fold macro F0.5 | US | India | Weighted |
|---|---|---|---|
| v7 (submitted) | 0.9832 | 0.9755 | 0.9790 |
| v10 without the cross-encoder | 0.9851 | 0.9820 | 0.9834 |
| **v10 complete** | **0.9882** | **0.9866** | **0.9873** |

v10 − v7 = **+0.0083**, 95% CI [+0.0082, +0.0085] (paired cluster bootstrap, 1,000 draws);
fold-honest thresholds reproduce the same values. Remaining loss, in F0.5 points: candidates never
proposed 0.0058 (US) / 0.0079 (India), candidates scored too low 0.0048 / 0.0044, wrong merges
0.0011 / 0.0011 — against a perfect-scorer ceiling of 0.9942 / 0.9921.

## Repository layout

```
PROJECT_EXPLAINED.md          the full explainer — how and why, from zero
HANDOFF.md                    the complete project record
code/business_entity_resolution/
├── src/                      the pipeline (see PROJECT_EXPLAINED.md#file-map)
│   ├── normalize.py          name/address cleaning + Indian-script dictionary mining
│   ├── blocking.py           multi-key candidate search + learned ranking cascade
│   ├── metablock.py          supervised meta-blocking
│   ├── features.py featidf.py  the 66 pair features + 7 IDF-weighted overlaps
│   ├── train.py train_oof.py train_pruned.py   the two matchers and out-of-fold scoring
│   ├── xenc_export.py kaggle/  the cross-encoder: export of uncertain pairs + Kaggle GPU job
│   ├── stack.py              the collective stacker (writes the submission files)
│   ├── decode.py             one-owner rule + expected-F0.5 decoding
│   ├── evaluate.py loco.py   evaluation harness, leave-one-country-out study
│   └── run_all.py            runs the whole pipeline end to end
├── run_v10.sh run_v10_finish.sh   the exact commands behind the v10 artifacts
├── README.md                 per-stage commands, timings, code map
└── requirements.txt
submissions/final_v7/         the submitted v7 results + thresholds/report
submissions/final_v10/        the v10 results, thresholds, report, methodology document
cloud/                        AWS setup scripts for a bigger-machine run (not used in the end)
make_submission.py            builds the official submission zip, with an MD5 identity check
```

**Not in this repo:** the challenge dataset (organizer-provided, not ours to redistribute), the
intermediate pipeline cache (`work*/`, multi-GB and fully regenerable) and the submission zips /
`candidate_pairs.tsv` (over GitHub's size limits). See [.gitignore](.gitignore).

## Running it

```bash
pip install -r code/business_entity_resolution/requirements.txt   # Python 3.13
cd code/business_entity_resolution/src
python metric.py          # sanity check: prints "metric OK"
python run_all.py         # CPU pipeline end to end, ~9 h on a 12-thread / 31 GB laptop
```

Needs the official dataset locally (path via `ER_DATA_DIR`, default
`6ab10eb3b23ba_student_resource/student_resource/dataset/`). Every stage caches to `work/` and can
be run standalone — see [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md)
for the per-stage table and the GPU step for the cross-encoder. Run heavy stages one at a time.

## Fair play

No internet lookups, APIs, geocoding or external data anywhere in the code. The Indian-script
dictionary and every rule and threshold are learned or tuned from the provided training data only;
unlabeled test text is used solely for unsupervised statistics (word frequencies, IDF), never for
pseudo-labels. Models: LightGBM (MIT) and `jhu-clsp/mmBERT-small` (MIT, 140M parameters — the limit
is 8B), fine-tuned only on the provided training labels. Kaggle was used as GPU compute, not as a
data source.
