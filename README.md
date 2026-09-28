# Business Entity Resolution — team gemma_gals

**Amazon ML Challenge 2026.** Public leaderboard **F0.5 = 0.968**.
Team: Soumya Sharma (leader), Anjali Singh, Prisha Raj — BIT Mesra, Patna Campus.

For the *complete* write-up (official rules, full submission log, the methodology document as
submitted, every code-map entry, the AWS guide, and the post-deadline redesign plan) see
**[HANDOFF.md](HANDOFF.md)**. This README is the short version: what the problem is and how our
solution approaches it.

---

## The problem, in one paragraph

We get business records — name, address, country — from three independent sources (`S1`, `S2`,
`S3`), each noisy in its own way: typos, abbreviations, transliterated Indian-script names,
missing house numbers, reordered address parts. `S1` is a clean, deduplicated reference list of
1.7M businesses. For **every** `S1` business, we must list every `S2`/`S3` record that is the
*same real business* — zero, one, or many. Scale rules out comparing everything to everything:
1.7M × ~10M candidates is ~17 trillion pairs. The data also plants deliberate traps — "sibling"
businesses sharing a building with a different unit number — so a careless model that matches on
name and street alone gets punished hard, because the scoring metric (**F0.5**) weights precision
**2×** over recall: one wrong merge costs almost as much as two missed matches.

## Our approach, in one picture

```
raw .tsv files
     │
     ▼
① NORMALIZE            clean names/addresses; translate Indian-script names with a dictionary
     │                  MINED FROM THE TRAINING ANSWERS (not hand-written); expand abbreviations;
     │                  parse every number ("1577/15" → main 1577, sub 15)
     ▼
② BLOCKING              for each S1, ~300 plausible S2/S3 records via 8 kinds of cheap shared
     │                  keys (rare name words, house number + street word, ...); a LEARNED
     │                  two-stage LightGBM cascade ranks and keeps the top 15/source; then a
     │                  SUPERVISED META-BLOCKING ranker cuts that to ~5.3/entity            ──► candidate_pairs.tsv
     ▼
③ FEATURES              66 numbers per remaining pair: name/address similarity (rapidfuzz),
     │                  house-number structure agreement, shared/extra tokens, and — the
     │                  highest-signal group — RIVAL FEATURES: does a competing S1 fit this
     │                  record better than we do?
     ▼
④ MATCHING MODEL        LightGBM gives each pair a match probability. A SECOND matcher is
     │                  trained on exactly the ~5-candidate distribution the first one only sees
     │                  after meta-blocking (not the full ~28), and a COLLECTIVE SECOND STAGE
     │                  re-scores every pair using both matchers' predictions for the entity's
     │                  rival candidates and for records competing S1s also want — twice
     │                  (iterative collective classification)
     ▼
⑤ DECISION RULES         one S2/S3 record → at most ONE S1 (proven true in the data: 100% of
     │                  the time); keep the probability-sorted prefix that MAXIMIZES EXPECTED
     │                  F0.5 (not just "probability > 0.5"); say nothing when unsure — a
     │                  wrong guess costs far more than a miss; thresholds tuned per country,
     │                  France (never seen in training) uses the stricter setting
     ▼
matching_results.tsv  →  leaderboard
```

Every stage in bold above exists because we **measured** a property of the data first, not because
it's the textbook default. See [HANDOFF.md §3–5](HANDOFF.md) for the full evidence behind each
choice, including exact numbers (recall at every stage, error breakdowns, ablations we tried and
rejected).

## Why this design

- **Blocking sets the ceiling.** A true match blocking never proposes can't be recovered by any
  model downstream, so we invested in blocking as much as in the classifier: a hand-tuned
  multi-key search, *then* two learned re-ranking stages on top of it, taking candidates/entity
  from ~300 → 15 → 5.3 while pair recall only drops from 0.983 → 0.971 → 0.970.
- **F0.5 rewards being right, not being complete.** Everything downstream of blocking is built to
  say "no match" by default: the one-owner constraint, the expected-F0.5-optimal prefix (instead
  of a flat probability cutoff), and per-country thresholds tuned specifically to avoid false
  merges on the sibling-business traps the dataset plants on purpose.
- **The model has to transfer to an unseen country.** France appears only in the test set. No
  feature is country-specific (no one-hot country flag anywhere), and the France decision
  threshold was validated after the deadline with a leave-one-country-out study (train on one
  country, score the other) rather than left as a guess — see `code/business_entity_resolution/src/loco.py`
  and [HANDOFF.md §8](HANDOFF.md#8-post-deadline-redesign-plan-for-the-finale).
- **Collective, not just pairwise, decisions.** Two S1 businesses often compete for the same
  ambiguous S2/S3 record. Rather than scoring each `(S1, candidate)` pair in isolation, the
  collective second stage (`src/stack.py`) feeds each pair's score the probabilities of its
  *rivals* — what else is this record likely to be, and what else does this S1 likely already
  have — which is exactly the structure the sibling-business traps are designed to defeat.

## Results

| Version | What changed | Out-of-fold F0.5 | Leaderboard |
|---|---|---|---|
| v1 | LightGBM baseline + one-owner decoding | 0.9628 | — |
| v2 | Blocking key fixes (typo-proof rarity, wider key set) | 0.9718 | 0.963 |
| v4 | Learned two-stage blocking cascade | 0.9746 | 0.966 |
| v5 | + collective second-stage model | 0.9781 | 0.967 |
| v6 | + a second matcher trained on the pruned candidate distribution | 0.9789 | 0.968 |
| **v7 (final)** | + a second collective classification round | **0.9790** | **0.968** |

Out-of-fold numbers are macro F0.5 over all 2.2M training entities, weighted US 0.45 / India 0.55
to match the test country mix — never on data the scoring model saw during training. Full
breakdown (per-country, singleton accuracy, false-positive/negative taxonomy) in
[HANDOFF.md §5](HANDOFF.md).

## Repository layout

```
code/business_entity_resolution/
├── src/                  the pipeline — see HANDOFF.md §6 for a file-by-file map
│   ├── normalize.py      name/address cleaning + Indian-script dictionary mining
│   ├── blocking.py       multi-key candidate search + learned ranking cascade
│   ├── metablock.py      supervised meta-blocking (the second candidate-pruning stage)
│   ├── features.py       the 66 pairwise features
│   ├── train.py / train_pruned.py / train_oof.py   the two first-stage matchers
│   ├── stack.py          the collective second-stage model
│   ├── decode.py         one-owner rule + expected-F0.5-optimal decoding
│   ├── loco.py           post-deadline: leave-one-country-out validation
│   ├── featv2.py         post-deadline: sibling-trap (unit/suffix/digit) features
│   └── run_all.py        runs the whole pipeline end to end
├── requirements.txt
└── run_v9.sh              post-deadline experiment chain (§8 of HANDOFF.md)
cloud/                     AWS setup scripts for a bigger-machine run (not used in the final
                           submission — the laptop-only pipeline already reached 0.968)
submissions/final_v7/      the final submitted matching_results.tsv, plus its threshold/report JSON
make_submission.py         builds the official submission zip layout
HANDOFF.md                 the full project document (rules, plan, methodology, redesign plan)
```

**Not in this repo:** the challenge dataset (organizer-provided, not ours to redistribute), the
intermediate pipeline cache (`work/`, multi-GB and fully regenerable), and the submission zip/
`candidate_pairs.tsv` (over GitHub's size limits). See `.gitignore`.

## Running it

```bash
pip install -r code/business_entity_resolution/requirements.txt   # Python 3.13
cd code/business_entity_resolution/src
python run_all.py
```
Needs the official dataset locally (path set via `ER_DATA_DIR`, default
`6ab10eb3b23ba_student_resource/student_resource/dataset/`). CPU-only, ~8.5 h end to end on a
12-thread, 31 GB laptop; every stage caches to `work/` and can also be run standalone — see
[HANDOFF.md §6](HANDOFF.md#6-pipeline-code-how-to-run-it) for the per-stage command/output/timing
table.

## Fair play

No internet lookups, APIs, geocoding, or external data anywhere in the code. The Indian-script
dictionary and every rule/threshold are learned or tuned from the provided training data only;
unlabeled test text is used solely for unsupervised statistics (word frequencies), never for
pseudo-labels. The model is LightGBM (MIT license), well under the 8B-parameter limit.
