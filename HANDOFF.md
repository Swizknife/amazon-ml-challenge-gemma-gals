# HANDOFF: Amazon ML Challenge 2026 — Business Entity Resolution — team gemma_gals

**Team:** Soumya Sharma (leader), Anjali Singh, Prisha Raj — BIT Mesra, Patna Campus.
**Merged from 9 project markdown files into this single document on 28 Sep 2026.** The
originals (`START_HERE.md`, `PROBLEM_EXPLAINED.md`, `SOLUTION_PLAN.md`, `REDESIGN_PLAN.md`,
`richtext_converted_to_markdown.md`, `cloud/AWS_GUIDE_TEAM.md`, `cloud/AWS_STEPS.md`,
`code/business_entity_resolution/README.md`, `submission/Documentation_template.md`) were
deleted after merging; their content is preserved below, section by section.

**Contents**
1. [Final status](#1-final-status)
2. [The official problem statement](#2-the-official-problem-statement-amazon--unstop)
3. [The problem, explained plainly](#3-the-problem-explained-plainly)
4. [Solution plan and submission log](#4-solution-plan-and-submission-log)
5. [Methodology document (as submitted)](#5-methodology-document-as-submitted)
6. [Pipeline code: how to run it](#6-pipeline-code-how-to-run-it)
7. [AWS cloud guide](#7-aws-cloud-guide)
8. [Post-deadline redesign plan (for the finale)](#8-post-deadline-redesign-plan-for-the-finale)

---

## 1. Final status

- **Deadline:** 27 Sep 2026, 23:59 IST — passed.
- **Final upload:** `matching_results.tsv` from **v7** (a second collective-classification round on
  top of v6). Public leaderboard **0.968**. Zip submitted: `gemma_gals_submission2.zip`.
- **The zip's `output/matching_results.tsv` is byte-identical to `submissions/final_v7/matching_results.tsv`**
  (verified by hash before upload).
- **Validator:** ran clean — `PASS — no blocking issues found. Safe to submit.`
- **Folder cleanup (28 Sep):** everything except the dataset, the code, the best submission
  (`gemma_gals_submission2.zip` + `submissions/final_v7/`), `work_v4/` (kept for the finale
  redesign work), and this document was deleted — old `work/` cache (15 GB), 11 duplicate
  `output*` directories, 6 superseded submission folders, 2 unused zips, and 2 stale AWS bundle
  zips.
- **Leave-one-country-out (LOCO) check (28 Sep, post-deadline):** confirms the France threshold
  guess used at submission time (`t_single=0.65, t_match=0.3, miss_mass=0.34`) was already within
  0.0001 F0.5 of the best value the data supports. Full numbers in §8 and `work_v4/loco_report.json`.
  An unseen country costs roughly **0.048 F0.5 for India-like data and 0.012 for US-like data**
  versus training on it directly — the honest ceiling on how much a "France fix" can be worth.
- **Score progression:** v1 (not uploaded) → v2 **0.963** → v4 **0.966** → v5 **0.967** →
  v6 **0.968** → v7 **0.968 (final)**.

---

## 2. The official problem statement (Amazon / Unstop)

*(Verbatim content, syntax-highlighting menu boilerplate stripped.)*

### Business Entity Resolution Challenge

In large-scale commercial platforms, business identity data arrives from multiple independent
sources — each contributing partial, noisy fragments of information about the same real-world
entities. These fragments share no common identifiers, and the challenge of determining which
records refer to the same business is known as Entity Resolution (ER). The task: build an ML
solution that, given business records from 3 independent data sources with noisy and
inconsistent fields, determines which records across sources refer to the same real-world
business entity.

Source 1 is the deduplicated reference source. The task is to find all matching records from
Source 2 and Source 3 for each Source 1 entity. A Source 1 entity may match zero, one, or many
records from Source 2 and Source 3.

### File format
All files are tab-separated (.tsv), and submissions must be tab-separated too (addresses and ID
lists both contain commas). Read with an explicit tab separator; without `sep="\t"` pandas
silently produces one column.

### Data description
Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) has:
1. **entity_id** — unique id; prefix `S1-`/`S2-`/`S3-` indicates the source.
2. **business_name** — may contain abbreviations, legal suffixes, typos, transliterations.
3. **business_address** — may be partial, reformatted, missing components, landmark-based.
4. **country** — training covers US and India; **test additionally has France**, absent from
   training. Country is an open set of labels — do not hard-code, filter, or one-hot to
   {US, India}; every test entity, France included, must appear in the submission.

No separate *source* column — the source is given by the `entity_id` prefix and by which file it
appears in.

`train_ground_truth.tsv` has two columns: **source1_entity_id** and **matched_entity_ids**
(comma-separated Source 2/3 ids, empty when there are no matches).

**Noise patterns to expect:**
- Name: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix inconsistencies,
  DBA/trade names, punctuation (& vs "and"), word-order transpositions, typos.
- Address: abbreviations (Rd/Road, St/Street), transliteration variants, missing components (no
  PIN/state), landmark references (Near SBI ATM), municipal numbering formats, reordering.

### Dataset details
- **Training:** 3 sources with ground-truth matching labels.
- **Test:** 3 sources, no labels. Hold out a validation split from training and score it yourself
  with the F0.5 formula below.

### Output format
Two tab-separated files in `output/`:

1. **matching_results.tsv** — final matches. **The only file scored on the leaderboard** —
   uploaded to the Portal during the challenge.
   - Columns: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 ids, no quoting).
   - Every test S1 entity gets exactly one row; empty `matched_entity_ids` for singletons; no
     duplicate ids within a list; only S2/S3 ids that exist in the test set.
2. **candidate_pairs.tsv** — the candidate set from blocking, *before* the final matching model
   narrows it down. This is the exact set the model runs inference over — the last stage of a
   multi-stage blocking pipeline, not an earlier, later-filtered pass. Not scored, but used to
   analyse blocking quality (recall ceiling, reduction ratio) and to verify the pipeline. Every id
   in `matching_results.tsv` must appear here (a mismatch signals a pipeline bug).

**Validate before submitting** (stdlib only, no dependencies), from the `student_resource/`
folder:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Prints `PASS` (exit 0) when safe to submit, or a numbered issue list (exit 1). It only reads the
output and test source files; it does not compute the score.

### Final submission package
Beyond live leaderboard uploads, every team submits one zip used to reproduce results, audit
blocking, and check fair-play/model-license rules — top teams' packages are reviewed in detail
before final rankings are confirmed.

```
_submission.zip
├── output/
│   ├── matching_results.tsv        # same file uploaded to the leaderboard
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── README.md               # how to reproduce end-to-end
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled-in methodology write-up
```
- **code/business_entity_resolution/** must be self-contained and runnable: anyone should be able
  to regenerate both output files from the training/test data using only what's in this folder.
- **Methodology document** — fill in `Documentation_template.md` and drop it in the zip (`.md` or
  a PDF export is fine, no rename needed).

### Constraints
1. Format exactly as described; submissions that fail validation are not evaluated.
2. `matched_entity_ids` must only reference S2/S3 entities in the test set; self-matches to S1 or
   nonexistent ids are rejected.
3. Every S1 entity must appear; missing entities cause rejection.
4. Duplicate ids in any list, or duplicate `source1_entity_id` rows, cause rejection.
5. Final model must be MIT/Apache-2.0 licensed and ≤ 8B parameters.

### Evaluation criteria
**F_β Score (β = 0.5)** — precision-heavy, penalizes false merges more than missed matches.
```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```
Computed as a **macro-average**: F0.5 per Source 1 entity, then averaged over all entities in the
evaluation set. Singletons are included: a no-match entity scores 1.0 for a correctly empty list,
0.0 for any predicted match. F0.5 weights precision 2× over recall because, in real-world entity
resolution, merging two distinct businesses is more damaging than missing a link.

**Worked example:** predicted `[S2-00047, S2-00193, S3-00812]`, truth `[S2-00047, S3-00812]` →
Precision 2/3, Recall 1.0, **F0.5 = 0.714**.

### Leaderboard
- **Public** (during the challenge): a subset of test, live feedback.
- **Private** (revealed after the challenge): the rest of test — **this decides final rankings.**
- The same full-test predictions serve both; the split is applied during scoring.

### Submission requirements
1. **Leaderboard:** upload `matching_results.tsv`, tab-separated, exact column names.
2. **Final package:** the zip above — `output/` with both files, `code/business_entity_resolution/`,
   the methodology document. All teams submit it; top packages are reviewed before final rankings.
3. The methodology document must describe: methodology used; candidate generation/blocking
   strategy; model architecture and feature engineering; any other relevant approach details.
   No page limit — prioritise clarity and technical depth over brevity.

### Academic integrity and fair play
**Strictly prohibited — external data lookup:** commercial entity-resolution APIs/services,
government-registry lookups, geocoding APIs for address normalization, any external data
augmentation from internet sources. All approaches, methodologies and code pipelines are reviewed
and verified; evidence of external lookup means **immediate disqualification**. The challenge
tests ML/data-science skills using only the provided training data.

### Tips for success
Invest in blocking/candidate generation (it bounds recall); explore Jaccard/Levenshtein/TF-IDF
cosine string similarity; pay attention to country-specific address patterns; weigh precision over
recall per F0.5; don't neglect singletons (correct "no match" is worth a full 1.0); validate output
format before submitting.

### Guidelines
All registered teams can play; no negative marking; all eligibility/authenticity/final-judgement
decisions rest with Unstop and the organizer.

---

## 3. The problem, explained plainly

*(Our own no-prior-ML-knowledge write-up. Every number below comes from the actual dataset.)*

### TL;DR
1. **Three sources:** business records (name + address + country) from 3 separate sources. The
   same real business shows up several times, written differently each time.
2. **The task:** Source 1 is the clean "master list". For every Source 1 business, list which
   Source 2/Source 3 records are the same business — none, one, or many.
3. **Scoring:** F0.5 per business, averaged. Rewards being right more than being complete — a
   wrong match hurts more than a missed one.
4. **Scale:** 1.7M businesses against ~10M records — can't compare everything with everything.
   First a short list of likely candidates (blocking), then a model decides (matching).
5. **Deadline:** 27 Sep 2026, 23:59 IST. ~3–5 leaderboard uploads/day. Never use outside data,
   APIs or internet lookups — disqualification.

### The problem in plain words
Like phone contacts after syncing WhatsApp, Gmail and a SIM card producing `Rahul Sharma`,
`rahul s.`, `Rahul (Office)` — one person, three entries. Merging correctly is easy for a human,
hard for a computer, and you must **not** merge two different people (or businesses) who happen
to share a building. This is **Entity Resolution (ER)**: deciding which records refer to the same
real-world thing with no shared id, at ~12 million records.

### Why it matters
Amazon registers the same business many times: as a seller, an Amazon Business buyer, via a
supplier feed, a partner list. A **wrong merge** sends invoices, payments, tax filings or fraud
flags to the wrong company — expensive, sometimes illegal. A **missed merge** means duplicate
accounts and scattered history — annoying, less damaging. Hence the precision-heavy metric.

### The data
All files in `6ab10eb3b23ba_student_resource/student_resource/dataset/`, tab-separated.

| File | Rows | What it is |
|---|---:|---|
| `train/train_source1.tsv` | 2,206,821 | Master list (deduplicated): US + India |
| `train/train_source2.tsv` | 5,034,616 | Source 2 records |
| `train/train_source3.tsv` | 5,285,603 | Source 3 records |
| `train/train_ground_truth.tsv` | 2,206,821 | The answers: which S2/S3 records belong to each S1 |
| `test/test_source1.tsv` | 1,732,544 | Businesses to resolve: US + India + **France** |
| `test/test_source2.tsv` | 4,887,273 | Source 2 records (test) |
| `test/test_source3.tsv` | 5,082,316 | Source 3 records (test) |

Columns: `entity_id` (prefix tells the source), `business_name`, `business_address`, `country`
(`US`/`India` in train; test adds **France**, never in train). Ground truth: `source1_entity_id`,
`matched_entity_ids` (comma-separated, empty if no match).

**A real example — one business, seven records** (`S1-680951579`, "Shyam Consulting Pvt Ltd",
`1-D-206, Eksar Laxminarayan Chs., Eksar Road, Borivali West, Mumbai, Mumbai City, Maharashtra`):
all seven are the same business per ground truth — a Hindi-script name, a website-style name, the
junk word "Smt", the state written three ways (Maharashtra / महाराष्ट्र / MH), and an S2 address
missing the street.

**A second example — the name can be completely different:** "Kip K. Hannigan, Ph.D." at
8 Kelly Avenue, Marcellus, NY has a match named "Beloumbra" — a trade name (DBA, *doing business
as*). Only the address links it.

**A singleton example:** "Lucrum Automation Ltd", Belgaum, Karnataka has no matches — the correct
answer is an empty list.

**Key facts measured:**

| Fact | Value |
|---|---|
| Matches per S1 business | 0 to 11. Median 3, average ~3.5 |
| S1 businesses with no match ("singletons") | **5.6%** |
| Can one S2/S3 record belong to two S1 businesses? | **Never** — at most one owner |
| S2/S3 records that match nothing (distractors) | ~26% |
| Do true matches always share a country? | **Yes, 100%** |
| Test businesses in France (unseen in train) | 15% of test S1 |
| Test S2/S3 records per S1 vs train | ~24% more (a distribution shift) |

### The noise catalogue

**Name noise:** legal-suffix variations (`Pvt Ltd` ↔ `Private Limited` ↔ `Pvt-Ltd`; `PC` moved to
front); typos/garbled characters (`Meticulous Law Cchcambes Limited`, `8eth Synagouge`); random
accents (`Léarning`, `Ínvestment`); junk added (`<< Team Ecole`, `[Services]`, `(ID: 13882)`,
`Smt`, `Sri`, `M/s`); repeated words (`Hrs Investments Investments Pvt (Ltd)`); website-style names
(`shyamconsulting.com`); trade (DBA) names (`Beloumbra` = Kip K. Hannigan, Ph.D.); **Indian
scripts (9 of them)** — ~23% of India Source 2 matches have names in Hindi/Devanagari, Telugu,
Kannada, Tamil, Bengali, Gujarati, Malayalam, Odia or Gurmukhi.

**Good news about Indian scripts:** they're English business words written in Indian letters
(`कंसल्टिंग` = "consulting"). Only 1,498 distinct such words. Learned from training answers, this
dictionary covers **96.4%** of the Indian-script words in test.

**Address noise:** abbreviations (`Rd`/`Road`, `St`/`Street`; France `R.`/`Rue`, `RTE.`/`Route`);
Source 2 addresses often ALL CAPS; missing parts (**4.4%** of matched records have an empty
address); re-ordering; state written differently (`MH`/`Maharashtra`/`महाराष्ट्र`); number noise
(extra numbers, changed numbers `G-904`↔`G-906`, sub-numbers `No. 131`↔`No. 131/4` — only **73%**
of true matches share the first house number); landmarks (`Near Fortis Hospital`).

**Traps — look-alikes that are NOT the same business:** deliberately planted "sibling" businesses
— same building, similar name, different unit, but a different business. E.g. "Baba Balaji
Ventures, 1577/8, …" vs "Baba Balaji Ventures Agencies Limited, 1577/15, …" — not a match. **25.6%**
of non-matching records look similar to some S1; **36.6%** of singletons have such a look-alike
(**68.5%** in India). This is exactly where a careless model loses points.

**France — only in test:** 15% of test businesses, never seen during training. French names use
very generic words (Club, École, "& Fils", SARL), making France the riskiest country for wrong
matches.

### The task, precisely
For every business in `test_source1.tsv`, output one row (`source1_entity_id`,
`matched_entity_ids`); the list may be empty, one id, or many. Only S2/S3 ids that exist in test
are allowed. An S2/S3 record belongs to at most one S1 (used as a rule in the solution).

### Scoring, in plain words
**Precision** = of what you caught, how much is the right fish; **recall** = of all the right fish
in the lake, how many you caught. F0.5 weights precision 2× recall:
```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

| Prediction (true count = 3) | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| all 3 correct | 1.00 | 1.00 | **1.000** |
| 2 correct, missed 1 | 1.00 | 0.67 | **0.909** |
| 3 correct + 1 wrong | 0.75 | 1.00 | **0.789** |
| 1 correct, missed 2 | 1.00 | 0.33 | **0.714** |
| 2 correct + 1 wrong | 0.67 | 0.67 | **0.667** |
| 3 correct + 2 wrong | 0.60 | 1.00 | **0.652** |
| nothing | — | 0 | **0.000** |

**Lesson:** one wrong match costs almost as much as two misses. Singletons: predict empty → 1.0;
predict anything → 0.0. Score is a **macro average** — per business, then averaged, every business
counts equally.

**Public vs private leaderboard:** public scores a live subset; private (revealed at the end,
decides final ranking) scores the rest. Overfitting to the public board can hurt the private one —
trust your own validation instead.

### The big picture: how we solved it
1.7M × ~10M ≈ 17 trillion comparisons is ~200 days at a million/s — so work in stages:
```
raw .tsv → ① normalize (clean names/addresses, translate Indian scripts)
         → ② blocking (short list of ~20 plausible candidates per S1) → candidate_pairs.tsv
         → ③ features (name/address similarity, house-number agreement, rival fit)
         → ④ model (LightGBM: probability "same business")
         → ⑤ decision rules (one S2/S3 record → at most one S1; maximize expected F0.5;
                              empty list when unsure)
         → matching_results.tsv → leaderboard
```
Key ideas: training data teaches patterns ("same house number + similar name → usually a match");
features are the numbers a model sees, never raw text; LightGBM votes across hundreds of small
trees; a 20% holdout estimates the real score before spending an upload; thresholds turn
probabilities into decisions, tuned toward "yes" only when confident (F0.5); **blocking sets the
recall ceiling** — a missed candidate can never be recovered by the model.

### Rejection rules and fair play
Rejected when: a test S1 is missing or duplicated; the file isn't tab-separated or the header is
wrong; an `S1-` id or unknown prefix appears in a match list; an id repeats inside one list.
Always validate first (see §2). **Fair play:** allowed — the provided files, rules/dictionaries
learned from training data, hand-written abbreviation lists, MIT/Apache-2.0 models ≤ 8B params.
Not allowed — geocoding/maps APIs, registry lookups, commercial ER services or LLM APIs, any
internet-downloaded augmentation data. The team's own added rule: unlabeled test text is used only
for simple statistics (e.g. word frequencies), never to guess test labels for training.

### Gotchas
```python
import csv, pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t",
                 quoting=csv.QUOTE_NONE, keep_default_na=False,
                 dtype=str, encoding="utf-8", nrows=100_000)
```
Files are ~500 MB each — explore with `nrows=100_000` first. On Windows,
`set PYTHONIOENCODING=utf-8` before printing Hindi/Tamil text. Write with
`df.to_csv(path, sep="\t", index=False, encoding="utf-8")`.

### Glossary
Entity resolution (ER) · Record/entity · Singleton · Blocking · Candidate pair · Recall ceiling ·
Feature · Precision/recall · F0.5 · Macro average · False positive/negative · Holdout/validation ·
Overfitting · LightGBM · Threshold · Transliteration · TF-IDF/IDF · Levenshtein/Jaro-Winkler ·
One-owner rule · Distribution shift (test has ~24% more records/business, plus France).

---

## 4. Solution plan and submission log

### Approach in one picture
```
normalize → blocking (short list) → pair features → LightGBM → decision rules → matching_results.tsv
                   │                                                   │
                   └──► candidate_pairs.tsv                            └──► one-owner rule + F0.5-optimal lists
```

| Stage | What it does | Why it matters |
|---|---|---|
| **Normalize** | Cleans names/addresses; translates Indian-script names with a dictionary learned from training answers; expands abbreviations; parses every number | 23% of India S2 matches have Indian-script names; the dictionary covers 96.4% of test tokens |
| **Blocking** | Collects ~20 likely S2/S3 records per S1 via cheap rules: number+street word, rare name word, rare address-word pairs, website/phonetic keys | Can't compare 1.7M × 10M pairs; a match blocking misses is lost forever |
| **Features** | Name similarity scores, house-number agreement (main + sub), street/city match, extra distinctive words, "does another S1 fit this record better?" | The data has deliberate look-alike traps (same building, different business) |
| **Model** | LightGBM, match probability | Fast on CPU, no GPU needed |
| **Decisions** | One-owner rule; the list maximizing expected F0.5; empty when unsure; per-country thresholds | F0.5 punishes wrong matches; singletons score 0 on any wrong guess |

### What we verified in the data
- **Country:** 100% of true matches share country — blocking stays inside each country.
- **One owner:** an S2/S3 record never belongs to two S1s — the one-owner rule is safe.
- **House numbers:** only 73% of true matches share the first house number — several blocking
  keys are needed, not just the number.
- **Traps:** 25.6% of non-matches look like some S1; 68.5% of Indian singletons have a look-alike.
- **Train vs test:** test has ~24% more records per S1; singleton rate is similar (~6%). **France
  is riskiest** (21.8 look-alikes per S1) — strict France thresholds from the start.

### Timeline (IST, as planned)
| When | What | Upload |
|---|---|---|
| 26 Sep 15:00–16:00 | Setup, data loading, F0.5 scorer, 20% holdout, Indian-script dictionary | — |
| 16:00–19:30 | Normalizer v1, blocking v1 (+recall report), rule-based matcher, full test run | **Sub #1** |
| 19:30–23:30 | Features, LightGBM, decision rules | **Sub #2 before midnight** |
| 27 Sep 00:00–10:00 | Overnight: full OOF run, one-owner rule, threshold tuning, French rules | — |
| 10:00–14:00 | Fixes from error analysis | **Sub #3** + France-threshold probe |
| 14:00–18:00 | Final full run + validator | **Final sub** |
| 18:00–21:00 | Zip, 2-page document, README, requirements | Zip uploaded |

### Who did what
| Person | Owned |
|---|---|
| **Leader** (+ Claude writing code) | Pipeline, model, final runs, submissions |
| **Member A** | Abbreviation lists (US/India/France street types); holdout error-pattern analysis after Sub #2 |
| **Member B** | Quality, logbook, validator before every upload, methodology document, finale slides |

### Submission log
| # | Time (IST) | What changed | Holdout F0.5 (US/India) | Leaderboard | Notes |
|---|---|---|---|---|---|
| 1 | 26 Sep 16:30 | v1: LightGBM (58 features) + one-owner decoding | **0.9628** (0.9734/0.9471) | — | Blocking recall 93.3% |
| 2 | 26 Sep ~23:10 | v2: blocking fixes (DF≥2 keys, number+name keys, name-only keys), wider threshold grid | **0.9717** (0.9766/0.9645) | 0.963 (01:26) | Blocking recall 95.7% |
| 3 | not submitted | v3: + Model2Vec embeddings for fuzzy name retrieval | 0.9719 | — | +0.0002 only (noise) — kept as `ER_USE_EMB=1` flag, v2 stays |
| final (v2) | 27 Sep ~02:00 | v2 + supervised meta-blocking (28.6 → 4.72 candidates/S1) + pruned-OOF thresholds | OOF 0.9718 | **0.963** (02:16) | 4.72 candidates/S1 |
| **v4** | 27 Sep 11:32 | Learned blocking cascade (stage-0/stage-1 rankers, 15 candidates/source) + retrained matcher/meta-blocking/thresholds | Holdout 0.9755; pruned OOF US 0.9794/India 0.9706 | **0.966** (~12:30) | 6.55 candidates/S1 |
| **v5** | 27 Sep 19:20 | Collective second-stage model (`stack.py`): entity view, rival view, transitivity | pruned OOF US 0.9826/India 0.9744 (+0.0035) | **0.967** (~19:40) | |
| **v6** | 27 Sep 21:51 | Second matcher on the ~5 meta-blocked candidates/entity (`train_pruned.py`), both matchers stacked; France keeps stricter threshold | pruned OOF US 0.9831/India 0.9754 | **0.968** | |
| **v7 (final)** | 27 Sep 22:14 | v6 + a second collective round | weighted 0.97898 (+0.0001 over v6) | **0.968** | Adopted; uploaded as the final submission |

**Fair-play checklist:** no internet/APIs/geocoding/registry lookups anywhere in the code; only
the provided data, test text for word statistics only; only MIT/Apache/BSD/ISC libraries
(LightGBM); zip contains code that regenerates both output files from the data. All satisfied.

### Story for the document and the 7 Oct finale
1. **Cross-script dictionary learned from data** — handles India's 9 scripts, 96.4% word coverage.
2. **Constraint-aware decisions** — the one-owner rule and F0.5-optimal lists from a proven data
   property.
3. **Trap-aware matching** — built to reject sibling businesses in the same building.
4. **Ready for new countries** — country-agnostic features; France handled by a threshold plug-in.
5. **Business value** — confidence tiers (auto-merge/human review/reject), reason codes, runtime
   on a single laptop.

---

## 5. Methodology document (as submitted)

*(The filled-in `Documentation_template.md`, submitted inside `gemma_gals_submission2.zip`.)*

**Team Name:** gemma_gals · **Members:** Soumya Sharma (leader), Anjali Singh, Prisha Raj (BIT
Mesra, Patna Campus) · **Submission date:** 27 September 2026

### Executive summary
A CPU-only pipeline: (1) multilingual normalization with a cross-script dictionary learned from
training matches; (2) learned candidate generation — multi-key blocking (~300 key matches/entity)
ranked by a two-stage LightGBM cascade, then supervised meta-blocking down to ~5.3
candidates/entity (97.0% of true pairs survive); (3) a LightGBM matcher with 66 features; (4) a
second matcher trained on the trimmed candidates it actually scores, plus a collective
second-stage model re-scoring every pair using both matchers' predictions for rival entities and
co-matched records; (5) constraint-aware decoding picking each entity's match list by highest
expected F0.5, under the data-proven one-owner rule.

Pruned out-of-fold macro F0.5 over all 2.2M training entities: **US 0.9831, India 0.9754** (v6).
Public leaderboard: 0.963 (v2) → 0.966 (v4) → 0.967 (v5) → 0.968 (v6) → **0.968 (v7, final upload)**.

### Problem analysis (EDA findings)
- 2.2M S1, 10.3M S2/S3 in train; S1 has 0–11 matches (median 3), 5.6% singletons; no S2/S3 record
  belongs to two S1s; every true pair shares country.
- Only 73% of true pairs share the first house number; 4.4% of matched records have empty
  address; noise inserts/changes/drops numbers.
- 22.7% of India-S2 matches have Indian-script names (word-by-word English renderings, 1.5k-token
  vocabulary); a training-mined dictionary covers 96.4% of test tokens.
- Planted "sibling" hard negatives: 25.6% of unmatched records look like some S1; 36.6% of
  singletons have a look-alike (68.5% in India).
- **Train→test shift (measured):** test has 23% more S2/S3 records per S1 in every country, so
  orphan records are ≈39% of test vs 26% of train, while true links/entity stay ≈3.5. Reproduced
  on train (remove 18% of other entities, re-score 150k held-out) — costs only **−0.0009** F0.5,
  so density explains little of the leaderboard gap. **The gap sits in France** (15% of test,
  unseen in train: +91% more uncertain pairs than train) **and India** (+30%). Record-level stats
  are otherwise identical (script share, empty addresses, missing house numbers).

### Solution strategy
Blocking + supervised meta-blocking + gradient-boosted matcher + constrained decoding. Core
innovation: a data-mined cross-script dictionary; rival/consistency features against sibling
traps; learned meta-blocking for a very small candidate set; one-owner, expected-F0.5 decoding
with per-country thresholds tuned out-of-fold.

### Candidate generation (blocking)
**Normalization:** Unicode/accent folding; Indian-script dictionary with ITRANS fallback;
canonical legal forms (US/India/France); junk/id/honorific removal; address abbreviations and
state codes; structured numbers (`1577/15` → 1577, 15).

**Stage 1 — multi-key blocking** (polars joins per country/source, capped blocks). Keys: address
number + rare address word; rare name-token pair and very rare single tokens; concatenated
(website-style) name; rare address-word pair; number + name token; full house-number structure +
word; moderate name tokens for records without an address number. "Rare" requires document
frequency ≥ 2, so typos never take key slots — this alone raised pair recall from 93.3% to 95.7%.

**Learned ranking cascade (v4).** The key-match union (~300/entity) holds 98.3% of true pairs.
Stage 0: LightGBM on per-kind key evidence, keeps 50/source. Stage 1: + 4 cheap string
similarities, keeps 15/source. At equal volume, pair recall rises from 0.9577 (hand-made
pre-score) to **0.9711**.

**Stage 2 — supervised meta-blocking.** A 300-tree LightGBM ranker keeps a candidate if
probability ≥ 0.01 (plus each entity's best). Inputs: blocking-derived signals only (pre-score,
shared-key count, key kinds, rank/relative score within-entity and among rivals) + 4 cheap
similarities.

| Candidate set (train) | Candidates/S1 | Pair recall |
|---|---|---|
| Key union (before ranking) | ~300 | 0.9832 |
| Stage 1, pre-score top 15 (v2) | 28.6 | 0.9577 |
| Stage 1, learned cascade top 15 (v4) | 28.6 | 0.9711 |
| **+ Stage 2 meta-blocking (submitted)** | **5.32** | **0.9698** |

On held-out entities, meta-blocking alone left F0.5 unchanged (0.9712 → 0.9712, measured with v2).
**Test candidate pairs:** 11,345,222 (**6.55/S1**; stage 1 alone gave 28.5), down from ~1.7×10¹³
possible pairs. `candidate_pairs.tsv` is exactly this set.

### Matching model
**66 features, no country indicator** (so the model transfers to unseen France): name (ratio,
token-set, token-sort, partial, Jaro-Winkler on core names; concatenated-name containment;
shared/extra token counts; rarest-token cross-presence; legal-form agreement); address (token-set/
ratio on full and word-only address; word Jaccard; rarest-word agreement; house-number structure —
main equal, full structure equal, number-set Jaccard, numeric distance); context (rank/relative
score within the entity's list; **rival signals** — rank among all entities competing for the same
candidate, the top-2 features by gain; consistency with the entity's best other candidate;
key-kind flags).

**Model:** LightGBM binary classifier (MIT), trained on 400k entities (11.4M pairs), 988 rounds.

**Second stage — collective stacking (v5).** A 400-tree LightGBM re-scores each pruned pair from
first-stage probabilities of related pairs (stacked collective classification, following top
Foursquare POI-matching solutions' two-level boosting and TransClean-style transitivity). 22
features: entity view (rank of p, best other p, gap, confident-candidate count, sum of p, rank
within source); record view (best p any *other* entity gives the same record, and the gap);
transitivity (name/address similarity and same house number vs the entity's two most probable
other candidates, plus their p). Trained 2-fold on first-stage OOF scores — no pair scored by a
model that saw its label. Pruned OOF improves US 0.9794/India 0.9706 → **US 0.9826/India 0.9744**
(+0.0035 weighted); leaderboard 0.966 → 0.967.

**Second first-stage opinion (v6, final).** The main matcher fits on ~28 blocking candidates/400k
entities but only ever scores ~5/entity after meta-blocking. `train_pruned.py` fits a second
LightGBM on exactly that distribution (11.7M meta-blocked pairs, all 2.2M entities, 2-fold, early
stopping ~1,250 rounds). Alone: 0.9756 weighted vs 0.9746. The stacker takes both probabilities as
features; thresholds re-tuned on a wider grid. Pruned OOF: **US 0.9831/India 0.9754** (weighted
0.9789 vs 0.9781 for v5). France (no labels) keeps v5's stricter setting (t_single 0.65) rather
than the looser US/India values — under F0.5 a false merge costs more than a miss.

**Final (v7): a second collective round** (iterative collective classification). Neighbour
features rebuilt from round-1 stacked probabilities; a second 2-fold stacker trained on them plus
the round-1 probability; US/India t_single re-tuned to 0.55 (France stays 0.65). Pruned OOF:
**US 0.9832/India 0.9755** (weighted 0.97898 vs 0.97887 for v6). *Tested, not adopted (+0.0001
each):* a larger stacker (1,200 trees); wider trees (255 leaves).

**Threshold selection:** after one-owner assignment, each entity keeps the probability-sorted
prefix maximizing expected F0.5. Three parameters/country (singleton gate, minimum probability,
missed-match prior) tuned on full-universe 2-fold OOF scores, all competing entities present, as
at test time. France uses the stricter tuned setting.

### Results and error analysis
- holdout (150k): **0.9755** (US 0.9793, India 0.9698); progression v1 0.9628 → v2 0.9717 → v4 0.9755
- pruned OOF, all 2.2M train: US 0.9794/India 0.9706 (0.9746 weighted by test country mix)
- + collective stage (v5): US 0.9826/India 0.9744
- + second matcher (v6): US 0.9831/India 0.9754
- + second collective round (v7, final): **US 0.9832/India 0.9755**
- leaderboard: v2 0.963 → v4 0.966 → v5 0.967 → v6 0.968 → **v7 0.968 (final)**
- one-owner rule adds +0.0004 over the full universe

**Where F0.5 is lost (v1 analysis):** blocking misses 0.015/0.040 (US/India), model misses 0.008,
wrong merges 0.005. Only 2% of singletons got a wrong guess.

**Error volume (150k holdout):** 2,959 false-positive pairs vs 33,823 false-negative pairs; 66% of
false negatives are blocking misses, 34% scored too low. This asymmetry is intended under F0.5.

**Common false positives (planted sibling traps):**
1. Same name + an extra word/legal form, same building, different unit: `Rohan Business, …G-2` vs
   `Rohan Business LLP, …G-6`; `Great Charities Inc` vs `Great Charities Westgate Inc`, same address.
2. Same name/street, house number off by one dropped/changed digit (`135 Maple St` vs `14 Maple
   St`, `272 Iowa St` vs `27 Iowa St`); true pairs carry the same digit noise, so inherently
   ambiguous.
3. Identical name, empty candidate address — name alone can't separate branches.

**Common false negatives:**
1. Random-string trade names (`DOVAZETATAVO`, `ECTOORBIZEPH`) linked only by address.
2. Empty addresses + heavy name noise (`Smt BANNARI INDUSTRIES GROUP PRIVATE`).
3. True candidates ranked below the top 15 of ~300 key matches — v4's cascade recovered most
   (0.9577 → 0.9711); the rest of the gap to the key union (0.9832) is the largest remaining loss.

### Conclusion
Data analysis drove every design choice: the cross-script dictionary, trap-aware rival features,
typo-proof blocking keys, one-owner decoding. Supervised meta-blocking shrank the candidate set by
83% at no accuracy cost. The pipeline runs end to end on one laptop, no GPU, fully reproducible,
and transfers to an unseen country (France) via country-agnostic features.

### Appendix A — code artefacts
`code/business_entity_resolution/src/run_all.py` runs normalize → blocking → matching model →
out-of-fold → meta-blocking → predict (~2.5 h, 12-thread laptop, no GPU). See §6 below for
per-stage commands and timings. Only the provided data is used.

### Appendix B — additional results
**Model2Vec experiment (MIT, 2025).** `potion-multilingual-128M` (128M-param static distillation
of bge-m3) for fuzzy name retrieval (FAISS IVF) + cosine features. Proprietary models (e.g. Jev)
excluded on license/external-service grounds. Lifted blocking recall on empty-address records by
5–7 pts (overall 95.72% → 96.14%); F0.5 gained only +0.0002 (recovered name-only pairs are
inherently ambiguous). Kept the simpler pipeline; stays behind `ER_USE_EMB=1`.

**Research-driven extensions (27 Sep).** Gated: adopted only if weighted OOF F0.5 (US 0.45/India
0.55, the test mix) improves ≥ 0.001.

| Idea (source) | Implementation | Outcome |
|---|---|---|
| Learned candidate ranking (supervised blocking) | v4 two-stage LightGBM cascade over the key union | **Adopted.** Recall 0.9577→0.9711; OOF +0.0046; LB 0.963→0.966 |
| Cross-encoders beat bi-encoders under shift ("Beyond Scale and Generation", arXiv 2607.24688, 2026); mmBERT (JHU, 2025, MIT) | `xenc.py`: fine-tunes `jhu-clsp/mmBERT-small` (140M) on unsure pairs, 2-fold, stacked | Implemented, smoke-tested. **Not adopted:** laptop CPU ~13 pairs/s, too slow for ~1M pairs. Needs GPU/AMX. |
| Stacked collective classification (top Foursquare solutions; TransClean transitivity, 2025) | `stack.py` | **Adopted (v5).** OOF +0.0035; LB 0.966→0.967 |
| Train on the deployment distribution; multiple first-stage opinions in the stacker | `train_pruned.py` + `stack.py --second` | **Adopted (v6).** OOF +0.0008 |
| Iterative collective classification | `stack.py --iterate` | **Adopted (v7, final).** OOF +0.0001 |
| Jev (TypeSafe, 2026), OKF (Google, 2026) | Reviewed | Not usable — Jev proprietary/API-only; OKF is a KB file format, not a model |
| Density shift between train/test | Simulated test's 39% orphan rate on train | Explains only −0.0009 of the gap; test-like retraining not pursued |

---

## 6. Pipeline code: how to run it

*(From `code/business_entity_resolution/README.md`.)* Produces `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the challenge data. Only the provided files — no external data,
APIs or lookups.

**Setup:** `pip install -r requirements.txt` (Python 3.13).

**Run end to end** (~8.5 h on a 12-thread, 31 GB laptop, no GPU):
```bash
cd src
python run_all.py
```
Default paths (relative to repo root): data `6ab10eb3b23ba_student_resource/student_resource/dataset/`;
intermediates `work/`; outputs `output/`. Override with `ER_DATA_DIR`, `ER_WORK_DIR`, `ER_OUT_DIR`.
**Heavy stages run one after another — never two in parallel on a 31 GB machine** (see §1's laptop
note; two heavy jobs at once once turned a 25-min step into ~5.8 h of swapping).

Each stage caches its result in `work/`, so it can also run alone:

| Step | Command | Output | Time |
|---|---|---|---|
| 1. Normalize | `python normalize.py` | `work/translit.json`, `work/norm/*.parquet` | 5 min |
| 2a. Cascade rankers | `python blocking.py stage0` then `stage1` | `work/model_stage{0,1}.txt` | 40 min |
| 2b. Blocking | `python blocking.py train test` | `work/cands/{train,test}.parquet` (~28.5/S1) + recall report | 2.5 h |
| 3. Matching model | `python train.py` | `work/model.txt`, `work/thresholds.json` | 31 min |
| 4. Out-of-fold | `python train_oof.py` | `work/feats_train/`, `work/oof_train.parquet`, `work/thresholds_oof.json` | 60 min |
| 5. Meta-blocking | `python metablock.py` | `work/model_meta.txt`, `work/thresholds_final.json` | 20 min |
| 6. Predict | `python predict.py` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` (~6.5/S1) | 20 min |
| 7. Second matcher | `python train_pruned.py` | `work/oof_pruned.parquet`, `work/scored_test_pruned.parquet`, `work/model_pruned{0,1}.txt` | 35 min |
| 8. Collective 2nd stage, 2 rounds (v7) | `python stack.py --second oof_pruned --primary second --wide-grid --strict-unseen --iterate`, copy `output_stack/*.tsv` to `output/` (`run_all.py` does both) | `output_stack/matching_results.tsv`, `work/model_stack{0,1}.txt`, `thresholds_stack.json` | 40 min |

Validate from `student_resource/`:
```bash
python utils/validate_submission.py --matching <repo>/output/matching_results.tsv \
    --candidate <repo>/output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

### Candidate generation in two stages
1. **Blocking** (`blocking.py`), polars joins on 8 key kinds (address number + rare address word;
   rare name-token pair/very rare single tokens; concatenated name; rare address-word pair;
   number + name token; full house-number structure + word; moderate name tokens for
   number-less records). Key frequencies come from the split's own text; every block is capped.
   The ~300-match union is ranked by the learned cascade: stage 0 (per-kind key evidence, top 50/
   source), stage 1 (+4 cheap similarities, top 15/source). Train pair recall at equal volume:
   0.9711 (cascade) vs 0.9577 (hand-made pre-score). Without the ranker files, blocking falls back
   to the pre-score.
2. **Supervised meta-blocking** (`metablock.py`): a light LightGBM ranker on blocking-derived
   signals (pre-score, shared keys, key kinds, rank among the entity's and among rival entities'
   candidates) + 4 cheap similarities. Keeps a candidate at ranker probability ≥ 0.01, plus each
   entity's best. On train: 28.6 → 5.3 candidates/S1, pair recall 0.9711 → 0.9698 (F0.5 unchanged
   on held-out entities with v2). `candidate_pairs.tsv` is exactly this set.

### Code map (`src/`)
| File | Purpose |
|---|---|
| `config.py` | Paths and constants |
| `data_io.py` | Safe TSV reading (no quoting, no NA conversion, row-count check) and output writing |
| `rules.py` | Hand-written normalization lists: legal forms, address abbreviations, state codes |
| `translit.py` | Indian-script → English dictionary mined from training matches, ITRANS fallback |
| `normalize.py` | Name/address normalization, parallel over all sources |
| `blocking.py` | Multi-key blocking + learned stage-0/stage-1 ranking cascade (`stage0`, `stage1` commands); prints union vs top-K recall and recall by address status |
| `metablock.py` | Supervised meta-blocking, thresholds tuned on pruned OOF scores |
| `features.py` | 66 pair features: rapidfuzz similarities, token/number/legal-form agreement, rank/rival/consistency context, key-kind flags |
| `train.py` | LightGBM matching model, holdout evaluation, per-country threshold tuning |
| `train_oof.py` | Full-universe 2-fold out-of-fold scoring, threshold tuning with one-owner decoding |
| `decode.py` | One-owner assignment and expected-F0.5-optimal list selection |
| `metric.py` | The challenge metric (macro F0.5); `python metric.py` runs its self-test |
| `predict.py` | Test inference: context → meta-blocking → matching model → decoding → the two TSVs |
| `run_all.py` | Runs every step in order |
| `redecode.py` | Re-applies a thresholds file to saved test scores, without re-scoring |
| `analysis.py` | Error analysis: holdout false positives/negatives, blocking misses |
| `stack.py` | Collective second-stage model: re-scores pruned pairs from first-stage probabilities of competing/co-matched records (entity view, rival view, transitivity); 2-fold on OOF scores; writes a submission only if weighted OOF F0.5 improves ≥ 0.001 |
| `train_pruned.py` | Second first-stage matcher (v6): 2-fold LightGBM on the meta-blocked candidates it actually scores (~11.7M pairs, all 2.2M entities); feeds `stack.py --second oof_pruned`. Now also supports `--extra`/`--tag` for the featv2 experiment (§8) |
| `train_full.py` | Matcher variant on the meta-blocked candidates of ~2.05M entities. Adopted only if it beats `model.txt` by ≥ 0.001 on identical pruned holdout pairs |
| `xenc.py` | Optional (needs GPU/AMX; `torch`, `transformers`): mmBERT-small (MIT) cross-encoder fine-tuned on borderline pairs, stacked with the matcher. Same ≥0.001 gate. Too slow on laptop CPU — not used in the submission |
| `embed.py` | Optional experiment (`ER_USE_EMB=1`), off in the final submission — Model2Vec embeddings for fuzzy name retrieval and cosine features. +0.42 pts blocking recall, +0.0002 F0.5 — not adopted |
| `loco.py` | **New (§8):** leave-one-country-out validation — what an unseen country costs, and the best decoding policy for it |
| `featv2.py` | **New (§8):** sibling-trap features (unit conflicts, house-number-suffix conflicts, dropped-digit detection, branch-word detection) |

**Licenses:** polars (MIT), rapidfuzz (MIT), LightGBM (MIT), numpy (BSD-3), indic-transliteration
(MIT). Both models (matcher, meta-blocking ranker) are LightGBM gradient-boosted tree ensembles.

---

## 7. AWS cloud guide

*(Merged from `cloud/AWS_GUIDE_TEAM.md`, the full beginner walkthrough, and `cloud/AWS_STEPS.md`,
its short-form original — both preserved since the team decided to skip AWS for the final
submission, but the guide stays for reproducibility and for a possible finale run.)*

**Status:** AWS was **skipped by team decision** for the 27 Sep submission — the laptop-only
pipeline (15 candidates/source, 400k training examples) reached 0.968 without it. This guide
remains for reference and for the post-deadline redesign work in §8, where a GPU instance is
actually useful (the cross-encoder stage, `xenc.py`).

### What AWS was for
The laptop has 31 GB RAM, which forces small settings (15 candidates/business, 400k training
examples). AWS rents a bigger machine (32 CPU cores, 256 GB RAM) to run the **same code** with
bigger settings: 25 candidates/business (fewer true matches lost before the model even sees them
— the biggest error source) and 3× more training examples. Optionally, a second stage fine-tunes a
small multilingual language model (**mmBERT-small**, MIT) to re-check pairs the main model is
unsure about.

| Item | Value |
|---|---|
| Machine type | `r7i.8xlarge` (32 vCPU, 256 GB RAM) |
| Region | Asia Pacific (Mumbai) `ap-south-1` |
| Price | $2.18/hour |
| Main run | ~4.5 h ≈ $10 |
| Language-model stage (auto-starts after) | +2.5 h ≈ +$5 |
| Disk (100 GB) | < $0.50/day |

> **Golden rule:** a running machine costs money every hour, even idle. **Terminate the instance**
> when done or if anything goes wrong (step 11).

### 0. What you need
The AWS account with the credits redeemed (or a teammate IAM user — see below); the file
`cloud_bundle.zip` from the leader *(note: this bundle was deleted in the 28 Sep cleanup as a
stale duplicate of `code/` and `cloud/`; regenerate it by zipping `code/business_entity_resolution/`
+ `cloud/` if a cloud run is done again)*; Windows PowerShell with `ssh -V` working; a work folder
(e.g. `C:\er`).

**For the account owner — create a teammate IAM user (3 min):** Console search **IAM** → **Users**
→ **Create user** → enter a name, tick **Provide user access to the AWS Management Console**, set
a password → **Attach policies directly** → tick **AmazonEC2FullAccess** and
**ServiceQuotasFullAccess** → **Create user**. Send the teammate the sign-in URL, username,
password. Budgets and credits (steps 1–2) must be done by the account owner.

### 1. Redeem the credits (account owner, 5 min)
Log in at console.aws.amazon.com → top-right account name → **Billing and Cost Management** →
**Credits** → **Redeem credit** → paste the promo code from the challenge email → **Redeem**.

### 2. Set a budget alarm first (account owner, 3 min)
**Billing and Cost Management** → **Budgets** → **Create budget** → **Use a template** →
**Monthly cost budget** → amount **40 USD**, team emails → **Create budget**. (Emails you if
spending exceeds expectations — e.g. a forgotten running machine.)

### 3. Choose the region (10 s)
Top-right of the console → **Asia Pacific (Mumbai) ap-south-1**. Every later step must happen in
this same region — if the machine seems to vanish later, check this first.

### 4. Check the vCPU quota (2 min)
Console search **Service Quotas** → **Amazon EC2** → search **"Running On-Demand Standard (A, C,
D, H, I, M, R, T, Z) instances"** → read **Applied quota value**:

| Quota | Machine | Notes |
|---|---|---|
| **≥ 32** | `r7i.8xlarge` | the plan |
| **16–31** | `r7i.4xlarge` (16 vCPU, 128 GB, $1.09/h) | slower (~6–7 h), still works |
| **< 16** | — | click **Request increase** to 32, tell the leader; approval can take a while |

### 5. Launch the machine (5 min)
EC2 → **Launch instance** → Name `er-bigrun` → AMI **Ubuntu Server 24.04 LTS (64-bit x86)** →
Instance type `r7i.8xlarge` (or per step 4; if unavailable try `r6i.8xlarge` then `r5.8xlarge`) →
Key pair: **Create new key pair**, name `er-key`, type RSA, format `.pem` → downloads
`er-key.pem`, move it to the work folder and keep it private → Network: **Allow SSH traffic from
My IP** (not "Anywhere") → Storage: change to **100 GiB gp3** → **Launch instance**. Wait for
**Running** + status checks passed (2/2 or 3/3, ~2 min), then copy the **Public IPv4 address**
(`<IP>` below).

### 6. Upload the code and connect (Windows PowerShell, 5 min)
```powershell
cd C:\er
icacls .\er-key.pem /inheritance:r
icacls .\er-key.pem /grant:r "$($env:USERNAME):(R)"
scp -i .\er-key.pem .\cloud_bundle.zip ubuntu@<IP>:~
ssh -i .\er-key.pem ubuntu@<IP>
```
The `icacls` lines lock the key file to the current user — SSH refuses a key others can read. On
`Are you sure you want to continue connecting (yes/no)?`, type `yes`. Once connected the prompt
looks like `ubuntu@ip-172-31-...:~$` — everything in step 7 is typed there.

### 7. Start the run (2 min)
Inside SSH:
```bash
sudo apt-get update -y && sudo apt-get install -y unzip tmux
unzip -o cloud_bundle.zip -d bundle
tmux new -s er                     # keeps running even if the laptop disconnects
```
**Normal machine (r7i.8xlarge):**
```bash
bash ~/bundle/cloud/run_aws.sh 2>&1 | tee ~/run.log
```
**Smaller machine (r7i.4xlarge, 128 GB):**
```bash
ER_N_TRAIN=800000 ER_OOF_FIT=600000 bash ~/bundle/cloud/run_aws.sh 2>&1 | tee ~/run.log
```
The script installs Python packages, downloads the official dataset, and runs the pipeline steps,
printing `>>> step NAME started HH:MM` for each. **Detach:** Ctrl+B, then D — the run continues
even if PowerShell is closed or the laptop is switched off.

**Then start the language-model stage** in a second tmux session (waits by itself for the main run
to finish, then runs ~2.5 h):
```bash
tmux new -s xenc
bash ~/bundle/cloud/run_xenc.sh 2>&1 | tee ~/xenc.log
```
Detach again (Ctrl+B, D). If it prints `No such file`, the bundle is older — see step 9.

### 8. Check progress anytime
```powershell
cd C:\er
ssh -i .\er-key.pem ubuntu@<IP>
```
| Command | Shows |
|---|---|
| `tail -n 20 ~/run.log` | Last 20 log lines |
| `grep ">>> step" ~/run.log` | Which steps started, and when |
| `tmux attach -t er` | The live window (Ctrl+B, D to leave again) |
| `free -g` / `df -h ~` | Memory / disk use |

Approximate timeline (r7i.8xlarge): setup+download+normalize 20 min; stage0+stage1 20 min;
blocking 60–90 min; train 40 min; oof 60 min; metablock 20 min; predict 20 min.

**Main run finished** when `~/run.log` ends with:
```
PASS — no blocking issues found. Safe to submit.
ALL DONE: results in /home/ubuntu/amazon-ml-challenge/output/
```
At that point, send the scores:
```bash
grep -E "F0.5|recall|candidates per S1" ~/run.log | tail -n 25
```
**Language-model stage finished:** `tail -n 5 ~/xenc.log` shows `XENC DONE` (~2.5 h after ALL
DONE). Its `weighted F0.5 ... (gain ...)` line says whether it helped — send that too. An error in
`xenc.log` isn't fatal: download (step 10) and terminate (step 11) anyway; the main results matter
most.

### 9. Only if step 7 said "No such file": older bundle
```powershell
cd C:\er
scp -i .\er-key.pem .\xenc_bundle.zip ubuntu@<IP>:~
ssh -i .\er-key.pem ubuntu@<IP>
```
```bash
unzip -o xenc_bundle.zip -d xbundle
tmux new -s xenc
bash ~/xbundle/cloud/run_xenc.sh 2>&1 | tee ~/xenc.log
```
Detach (Ctrl+B, D); finished when `~/xenc.log` ends with `XENC DONE`.

### 10. Download the results (5 min)
Inside the machine:
```bash
cd ~ && zip -r results_aws.zip amazon-ml-challenge/output amazon-ml-challenge/output_xenc run.log xenc.log
```
(`name not matched: ...output_xenc` warnings are fine if the optional stage wasn't run.) Then on
the laptop:
```powershell
cd C:\er
scp -i .\er-key.pem ubuntu@<IP>:~/results_aws.zip .
```
Send right away (e.g. WhatsApp): the last 40 lines of `run.log`, plus `output\thresholds_final.json`
from inside the zip — these small files carry the scores needed to decide. Send `results_aws.zip`
(~100–250 MB) via Drive; it contains `matching_results.tsv` and `candidate_pairs.tsv`.

### 11. TERMINATE the machine (the most important step)
Only after `results_aws.zip` is safely on the laptop — terminating deletes the machine and
everything on it. EC2 → **Instances** → tick `er-bigrun` → **Instance state** →
**Terminate (delete) instance** → confirm → wait for **Terminated**. Then EC2 → **Volumes**: the
list should be empty; if a 100 GiB volume remains, tick it → **Actions → Delete volume**. Next day,
check **Billing → Bills** to confirm charges were covered by credits.

### Troubleshooting
| Problem | Fix |
|---|---|
| `UNPROTECTED PRIVATE KEY FILE!` / `Permission denied (publickey)` | Re-run the two `icacls` lines; check `ubuntu@` (not your own name) and the right `<IP>` |
| `Connection timed out` | Internet IP changed — EC2 → instance → Security tab → security group → Edit inbound rules → SSH rule → Source **My IP** → Save. The run itself is unaffected |
| `VcpuLimitExceeded` at launch | Quota too low — see step 4 |
| "Instance type not supported in this Availability Zone" | Network settings → Edit → pick another Subnet, or use an alternative type from step 5 |
| IP changed after stop/start | Normal — copy the new Public IPv4. Prefer never "stopping"; leave it running until done |
| `Killed` / `MemoryError` in the log | Out of memory — resume with smaller settings: `ER_N_TRAIN=800000 ER_OOF_FIT=600000 START_AT=train bash ~/bundle/cloud/run_aws.sh 2>&1 \| tee -a ~/run.log` (inside tmux) |
| Log stopped with a `Traceback` | Share the last 40 lines (`tail -n 40 ~/run.log`); resume from the failed step instead of restarting: `START_AT=<step> bash ~/bundle/cloud/run_aws.sh 2>&1 \| tee -a ~/run.log`. Steps: `normalize stage0 stage1 blocking train oof metablock predict` |
| `No space left on device` | The 100 GiB disk step was skipped — download what's needed, terminate, relaunch with 100 GiB |
| PowerShell closed during the run | No problem — tmux keeps it running; reconnect and `tmux attach -t er` |
| Nothing works and it's late | Download what exists (step 10), terminate (step 11), report — the laptop result still stands |

### Checklist
- [ ] Credits redeemed, $40 budget alarm set
- [ ] Region = Mumbai
- [ ] Machine launched with 100 GiB, SSH from My IP
- [ ] Main run started in tmux `er`; language-model stage started in tmux `xenc`
- [ ] `PASS` and `ALL DONE` in `run.log`; scores reported
- [ ] `XENC DONE` in `xenc.log` (or the deadline reached)
- [ ] `results_aws.zip` downloaded; scores reported
- [ ] Instance **terminated**, Volumes list empty

---

## 8. Post-deadline redesign plan (for the finale)

*(From `REDESIGN_PLAN.md`, written 27 Sep 23:30 IST, after the final upload window closed.
Nothing in this section changed the submitted result. Progress since then is folded in.)*

**Purpose:** what to change next, in priority order, with the evidence for each change, for the
7 Oct finale if gemma_gals makes the top 10, or for next year.

### Reality check first

| Flag | Evidence | What it means |
|---|---|---|
| "+2 points" (0.968 → 0.988) is not reachable | Our own training score (pruned OOF) is 0.979; the leaderboard is 0.968 | Closing the *whole* train→test gap gives about **+1.1 points**. **Realistic stretch target: 0.975–0.980.** |
| The gap is most likely France | If US/India score on test what they score in OOF (0.983/0.975, 38%/47% of test), France (15%) must be scoring about **0.90** to give 0.968 | France is the single biggest lever. This assumes US/India transfer perfectly, so 0.90 is a floor, not a fact |
| France is not short of matches | v6 predicts 3.42 matches/France S1 (US 3.36, India 3.29); France has 5.3% empty lists (US 5.7%, India 6.0%); France gets 8.4 candidates/S1 vs ~6.2 elsewhere | France errors are probably **wrong merges** (look-alikes: 21.8/France S1), not misses — fix precision, not recall |
| French rules already exist | `rules.py` already has SARL/SAS/SASU/EURL, rue/boulevard/chemin/impasse, bis/ter, CEDEX, apt/suite/unit/floor | "Add French abbreviations" is **not** a new lever |
| Blocking gains convert poorly | Blocking causes 66% of missed pairs, but Model2Vec's +0.4 pt recall gave only +0.0002 F0.5 | Recovered pairs are the hardest ones. Expect small F0.5 per point of recall |
| The France threshold (0.65) was a guess | France has no labels, so it was never measured — **now measured (see below)** | Replace the guess with a leave-one-country-out measurement |
| Other teams' repos are public | GitHub has ≥2 repos for this exact challenge | **Never open or copy them** — top packages are reviewed; copied code risks disqualification |

### Target architecture
```
                       ┌─ multi-key blocking (current) ─────────┐
normalize + parse ───► │                                         ├─► learned cascade (top 25) ─► meta-blocking
(unit / number fields) └─ contrastive bi-encoder ANN (future) ────┘                                   │
                                                                                                       ▼
          LightGBM matcher (66 + featv2 features)  +  cross-encoder on borderline pairs (future, GPU)
                                                                                                       │
                                                                                                       ▼
                       collective stacker (current) ─► per-country calibration (future) ─► one-owner + expected-F0.5 decoding
                                                                                                       │
                                                                                                       ▼
                               checks: OOF + leave-one-country-out + slice metrics + validator
```
The skeleton stays (blocking → matcher → collective stacker → constrained decoding); changes
target **transfer to an unseen country** and **the hardest pairs**.

### Changes, in priority order

**1. Leave-one-country-out (LOCO) validation — done, 28 Sep.** `code/business_entity_resolution/src/loco.py`
trains on US-only and scores India, and the reverse, to measure what an unseen country costs, and
to find the best decoding policy for one. Result (`work_v4/loco_report.json`; a fast vectorized
scorer was checked against `decode.py`+`metric.py` and matched to 1e-6):

| Probability used | Best mean LOCO F0.5 (both directions) | Current France policy (0.65/0.3/0.34) |
|---|---|---|
| Pruned matcher alone (`p_loco`) | 0.94600 at (0.75, 0.2, 0.17) | 0.94585 |
| + collective stacker, LOCO-fitted (`p_stack`) | **0.95355** at (0.55, 0.2, 0.17) | 0.95345 |

**Finding: the France threshold guess was already almost optimal** (within 0.0001 of the best
value the data supports, either way). Cost of an unseen country vs training on it directly:
**India-like ≈ −0.048 F0.5, US-like ≈ −0.012 F0.5** — the honest ceiling on how much any
"France-specific fix" can be worth. This took ~16 h (both directions), far longer than the ~40 min
estimated — LightGBM on the full 11.7M-pair pruned set, twice, dominates the time; a future run
should sample entities (`--sample-s1`) for faster iteration and only run the full version once at
the end.

**2. Sibling-trap features — built, not yet trained/evaluated.**
`code/business_entity_resolution/src/featv2.py` adds 14 features targeting the documented false
positives (§5): unit/shop/suite conflicts (`u_n1/u_n2/u_eq/u_conf`), house-number letter-suffix
conflicts including French `bis`/`ter`/`quater` mapped to letters (`hs_1/hs_2/hs_eq/hs_conf`),
dropped-digit detection (`h_lev`, `h_pre`), and rare extra-name-token/branch-word detection
(`x1_df/x2_df/x1_in_a2/x2_in_a1`). Measured prevalence: units in 20% of India addresses (~0 in
US); `bis`/`ter` in 3.8% of France addresses (~0 in train — exactly why LOCO, not train/holdout,
is the right way to validate this feature); letter suffixes in 2.4% (US)/10% (India). Unit-tested
on three handmade look-alike pairs (`17 BIS` vs `17`, `SHOP G-4` vs `G-6`, `272` vs `27 IOWA ST`)
— all three flagged correctly. `train_pruned.py` gained `--extra`/`--tag` options to train with
these features added; `run_v9.sh` chains `featv2.py → loco.py --extra → train_pruned.py --extra →
stack.py`. **Not yet run end-to-end or gated against the ≥0.001 rule** — do this first when
resuming.

**3. Cross-encoder on borderline pairs (needs a GPU).** `xenc.py` already exists and was
smoke-tested (see §5/§6) but is too slow on the laptop CPU (~13 pairs/s for ~1M borderline pairs).
Evidence for the direction: Ditto (VLDB 2021) showed pretrained transformers beat feature-based
matchers by large margins on dirty text; a 2026 study of 1,215 fine-tuning runs found
cross-encoders beat bi-encoders consistently, generative models help mainly under distribution
shift, and bigger models learn more shortcuts — relevant because France *is* a distribution shift.
This is where AWS is actually useful (§7): a GPU instance (e.g. `g5.xlarge`/`g6.xlarge`), 2–4 GPU
hours. Add Ditto-style markers around numbers/units (`[NUM] 135`, `[UNIT] G-2`) and augment by
token dropping, digit noise, span shuffling.

**4. Contrastive bi-encoder blocking (recall).** SC-Block (ESWC 2024): supervised contrastive
learning on matched/non-matched training pairs, then nearest-neighbour search — missed 43–45%
fewer pairs than self-supervised blockers in their benchmarks, with pipelines running 1.5–4×
faster at no F1 loss. Our version: train on the 2.2M S1 entities' true pairs with blocking-derived
hard negatives, add FAISS top-k to the key union, raise the cascade from 15 to 25/source (as
planned for AWS in §7). Targets the gap between the key union (0.9832 recall) and what's actually
kept (0.9698) — but recovered pairs are historically hard to convert into F0.5 (see the Model2Vec
result above), so expect a modest return.

**5. Per-country calibration before decoding.** Isotonic calibration per country on OOF scores;
France uses LOCO-derived calibration. Expected-F0.5 decoding is only optimal with calibrated
probabilities, and France gets 35% more candidates/S1 than elsewhere, which likely shifts the
rival features and probabilities. Cheap (minutes); do alongside item 2.

**6. Optional — needs a team rules decision before use:**
- **Synthetic French training pairs:** rewrite a sample of US training pairs into French surface
  forms (street type, legal form, `12 bis`, "rue de la X" word order) while keeping their labels,
  using only training data + a hand-written word map (like the existing abbreviation lists).
- **Adversarial validation on test features:** train a classifier to separate train pairs from
  France test pairs using features only (no labels), to find which features drift most.
- Both are arguably inside the team's "test text for unsupervised statistics only" rule, but
  **confirm against the official challenge rules (§2) before using either.**

### Checks (run before adopting any change)
1. Entity-level folds, full-universe OOF — current protocol, keep it.
2. **LOCO score next to OOF** (now available via `loco.py`). Adopt only if both improve, or OOF
   improves and LOCO doesn't fall.
3. Adoption gate: ≥ +0.001 weighted OOF (current rule; v7 was +0.0001 and tied v6 on the
   leaderboard — below-gate changes are noise).
4. Slice metrics: singletons, empty address, Indian script, number noise, sibling traps, per
   country.
5. Leaderboard-gap tracker: log OOF vs leaderboard for every upload; a widening gap signals
   overfitting to train.
6. Validator PASS, and the zip's `matching_results.tsv` hash-matched to the last leaderboard
   upload.

### What Amazon cares about (for the document and the finale pitch)
- **Precision first** — F0.5 reflects that a wrong merge of two businesses is worse than a missed
  link.
- **New-marketplace cold start** — France *is* Amazon's real problem when launching a marketplace
  with no labels yet. LOCO validation + a multilingual cross-encoder is the answer story.
- **Human-in-the-loop tiers** — auto-merge above a high threshold, human review in the borderline
  band, reject below it; reason codes from top features (same unit, number off by one digit, …).
- **Incremental ER and cluster health** — Amazon Science uses graph neural networks to detect
  *inconsistent clusters* in mature catalogs (bundles, substitutes). The collective stacker and
  one-owner rule are a lightweight version of the same idea.
- **Managed option** — AWS Entity Resolution is the managed product for this job. Not allowed in
  the challenge (external service), but worth naming as the production path.

### Honest estimate

| Change | Where it helps | Estimated gain | Status |
|---|---|---|---|
| LOCO + France policy | France (leaderboard) | +0.2 to +0.5 LB | **Done** — confirmed the existing policy is ~optimal, so the gain here is closer to 0 than hoped |
| Sibling-trap features | All countries | +0.1 to +0.3 OOF | Built, not yet trained/evaluated |
| Cross-encoder (GPU) | Borderline pairs, France | +0.2 to +0.5 OOF | Code exists (`xenc.py`), needs a GPU run |
| Contrastive blocking | Missed pairs | +0.1 to +0.3 OOF | Not started |
| Calibration | Decoding | small | Not started |

These are estimates from measurements and the literature, not yet fully measured on this data.
Each change must pass the checks above before being adopted.

### Sources
- [Ditto: Deep Entity Matching with Pre-Trained Language Models (VLDB 2021)](https://arxiv.org/abs/2004.00584)
- [SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines](https://arxiv.org/abs/2303.03132)
- [Beyond Scale and Generation: Understanding Language Model-based Entity Matching (2026)](https://arxiv.org/abs/2607.24688)
- [BEACON: Budget-Aware Entity Matching Across Domains (2026)](https://arxiv.org/abs/2603.11391)
- [Entity Matching using Large Language Models (Peeters et al., EDBT 2025)](https://arxiv.org/pdf/2310.11244)
- [Amazon Science: GNNs for Inconsistent Cluster Detection in Incremental Entity Resolution](https://www.amazon.science/publications/graph-neural-networks-for-inconsistent-cluster-detection-in-incremental-entity-resolution)
- [AWS Entity Resolution](https://aws.amazon.com/entity-resolution/)
- Public repos for this challenge (seen in search results, not opened): jakkulaayushpreetham/amazon-business-entity-resolution, Sanyam2005/amazonMl
