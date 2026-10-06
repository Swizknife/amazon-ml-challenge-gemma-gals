# How this project works — explained from zero

This document teaches the whole solution to someone who has **never built a machine-learning
system before**. No prior ML knowledge is assumed: every term is defined the first time it is
used, every design choice is justified, and every number comes from the actual runs in this
repository.

If you want the short version, read [README.md](README.md). If you want the official write-up as
submitted, read [submissions/final_v10/Documentation_template.md](submissions/final_v10/Documentation_template.md).
If you want the complete project history (rules, logs, plans), read [HANDOFF.md](HANDOFF.md).

**How to read this:** the sections follow the order the data flows through the pipeline. Each one
answers three questions — *what is this step*, *why is it needed*, *how is it done here* — and ends
with the files you can open to see it. Skipping around is fine; the glossary at the end holds every
term.

---

## Table of contents

1. [The problem in plain words](#1-the-problem-in-plain-words)
2. [The data, and what makes it hard](#2-the-data-and-what-makes-it-hard)
3. [The score, and why it dictates the whole design](#3-the-score-and-why-it-dictates-the-whole-design)
4. [The pipeline at a glance](#4-the-pipeline-at-a-glance)
5. [Step 1 — Reading the files without corrupting them](#step-1--reading-the-files-without-corrupting-them)
6. [Step 2 — Normalization: making two spellings look alike](#step-2--normalization-making-two-spellings-look-alike)
7. [Step 3 — Blocking: from 17 trillion pairs to 28 per business](#step-3--blocking-from-17-trillion-pairs-to-28-per-business)
8. [Step 4 — Meta-blocking: a model that shortens the shortlist](#step-4--meta-blocking-a-model-that-shortens-the-shortlist)
9. [Step 5 — Features: turning two records into numbers](#step-5--features-turning-two-records-into-numbers)
10. [Step 6 — The matcher: what LightGBM actually does](#step-6--the-matcher-what-lightgbm-actually-does)
11. [Step 7 — A second matcher, trained on the right distribution](#step-7--a-second-matcher-trained-on-the-right-distribution)
12. [Step 8 — The cross-encoder: a small transformer for the hard cases](#step-8--the-cross-encoder-a-small-transformer-for-the-hard-cases)
13. [Step 9 — The collective stacker: decisions that look at each other](#step-9--the-collective-stacker-decisions-that-look-at-each-other)
14. [Step 10 — Decoding: from probabilities to an answer sheet](#step-10--decoding-from-probabilities-to-an-answer-sheet)
15. [Step 11 — How we measured everything (and avoided fooling ourselves)](#step-11--how-we-measured-everything-and-avoided-fooling-ourselves)
16. [Results, stated honestly](#results-stated-honestly)
17. [Things we tried that did not work](#things-we-tried-that-did-not-work)
18. [Engineering lessons](#engineering-lessons)
19. [How to run it yourself](#how-to-run-it-yourself)
20. [File map](#file-map)
21. [Glossary](#glossary)

---

## 1. The problem in plain words

Imagine your phone's contact list after it has synced WhatsApp, Gmail and your SIM card. You see:

```
Rahul Sharma        rahul s.        Rahul (Office)
```

Three entries, one person. A human merges them in a second. A computer cannot, because there is no
shared ID anywhere — only text that looks *similar*. Deciding which records describe the same
real-world thing is a classic task called **entity resolution** (ER). It is also known as record
linkage, deduplication or identity resolution.

This challenge is entity resolution over businesses. We are given three lists of business records,
each with a name, an address and a country:

- **Source 1 (S1)** — a clean, already-deduplicated master list. Each row is one real business.
- **Source 2 (S2)** and **Source 3 (S3)** — messy lists from other systems. The same business may
  appear in both, spelled differently, or not at all. Many rows belong to no S1 business at all.

**The task:** for every single business in S1, output the list of S2/S3 records that are the *same*
business. The list may be empty (the business appears nowhere else), may have one entry, or may
have many.

**Why a company cares.** Amazon registers the same business repeatedly — as a seller, as an Amazon
Business buyer, through a supplier feed, in a partner list. If you *wrongly merge* two different
companies, invoices, payments, tax filings or fraud flags go to the wrong company: expensive and
sometimes illegal. If you *miss* a merge, you get duplicate accounts and split history: annoying
but survivable. That asymmetry is baked into the scoring, as section 3 shows, and it is the single
most important fact about this project.

### A real example from the data

`S1-680951579` — **Shyam Consulting Pvt Ltd**, *1-D-206, Eksar Laxminarayan Chs., Eksar Road,
Borivali West, Mumbai, Mumbai City, Maharashtra*.

The ground truth says **seven** records are this same business. Among them: the name written in
Devanagari script (`श्याम कंसल्टिंग प्रा. लि.`), a website-style name (`shyamconsulting.com`), a
version with the honorific `Smt` glued in front, the state written three different ways
(*Maharashtra* / *महाराष्ट्र* / *MH*), and one record whose address is missing the street entirely.

And a nastier one: **"Kip K. Hannigan, Ph.D."** at *8 Kelly Avenue, Marcellus, NY* has a true match
named **"Beloumbra"**. Nothing in the names is similar — it is a trade name (a "doing business as"
name). Only the address links them. Any approach that relies on names alone loses this pair.

---

## 2. The data, and what makes it hard

### Sizes

| File | Rows |
|---|---:|
| `train/train_source1.tsv` | 2,206,821 |
| `train/train_source2.tsv` | 5,034,616 |
| `train/train_source3.tsv` | 5,285,603 |
| `train/train_ground_truth.tsv` | 2,206,821 (the answers, for training only) |
| `test/test_source1.tsv` | 1,732,544 |
| `test/test_source2.tsv` | 4,887,273 |
| `test/test_source3.tsv` | 5,082,316 |

Training comes with the answers; test does not — that is what the leaderboard scores.

### The brute-force number

1.7M test businesses × ~10M test records ≈ **17 trillion** possible pairs. At a very optimistic one
million comparisons per second that is roughly 200 days of computation, and we have a laptop and a
deadline. **This single number is why the pipeline has the shape it has:** almost every design
decision below exists to avoid looking at pairs that cannot possibly match.

### Facts we measured before writing any model

Measuring the data *before* choosing a method is the habit that matters most in applied ML. These
came out of simple counting scripts over the training labels:

| Question | Answer |
|---|---|
| How many matches does an S1 business have? | 0 to 11. Median 3, average ~3.5 |
| How many have **no** match ("singletons")? | **5.6%** |
| Can one S2/S3 record belong to **two** S1 businesses? | **Never** — not once in 2.2M entities |
| Do true matches always share the same country? | **Yes, 100%** |
| S2/S3 records that match nothing (distractors) | ~26% in train, ~39% in test |
| France in test? | 15% of test businesses, and France **never appears in training** |

Two of these became hard rules in the code: we only ever compare records within the same country
(Step 3), and each record is awarded to at most one business (Step 10). A rule you *verified* on
2.2M rows is worth more than any model.

### The noise catalogue

**Names:** legal-form variants (`Pvt Ltd` / `Private Limited` / `Pvt-Ltd`); typos and garbled
characters (`Meticulous Law Cchcambes Limited`, `8eth Synagouge`); stray accents (`Léarning`);
injected junk (`<< Team Ecole`, `[Services]`, `(ID: 13882)`, `Smt`, `Sri`, `M/s`); duplicated words
(`Hrs Investments Investments Pvt (Ltd)`); website names (`shyamconsulting.com`); trade names that
share nothing with the legal name; and **nine Indian scripts** — 22.7% of India's S2 matches have
names in Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Odia or Gurmukhi.

**Addresses:** abbreviations (`Rd`/`Road`, French `R.`/`Rue`); ALL-CAPS text; missing parts (4.4% of
matched records have an **empty** address); reordered fields; states written three ways; and number
noise — only **73%** of true pairs share the same first house number (`No. 131` vs `No. 131/4`,
`G-904` vs `G-906`).

**The traps.** The dataset deliberately plants **sibling businesses**: same building, near-identical
name, different unit — and *not* a match.

```
"Baba Balaji Ventures,                  1577/8,  ..."  ─┐  NOT a match
"Baba Balaji Ventures Agencies Limited, 1577/15, ..."  ─┘
```

25.6% of non-matching records look similar to *some* S1 business; 36.6% of singletons have such a
look-alike (68.5% in India). This is where a naive "same name + same street ⇒ match" rule bleeds
points, and it is why so much of this solution is about *refusing* to match.

---

## 3. The score, and why it dictates the whole design

### Precision and recall, in one breath

You are fishing. You keep 10 fish.

- **Precision** = of the 10 you kept, how many were the right species? (Did you make mistakes?)
- **Recall** = of all the right-species fish in the lake, how many did you land? (Did you miss any?)

A model that predicts nothing has perfect precision and zero recall. A model that predicts
everything has perfect recall and terrible precision. Any useful score combines both.

### F0.5

The usual combination is the **F1 score**, which weights precision and recall equally. This
challenge uses **F0.5**, which weights **precision twice as heavily as recall**:

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Here is what that means for a business whose true answer has 3 matches:

| Our prediction | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| all 3 correct | 1.00 | 1.00 | **1.000** |
| 2 correct, missed 1 | 1.00 | 0.67 | **0.909** |
| 3 correct **+ 1 wrong** | 0.75 | 1.00 | **0.789** |
| 1 correct, missed 2 | 1.00 | 0.33 | **0.714** |
| nothing at all | — | 0 | **0.000** |

Read the third and fourth rows together: **one wrong match (0.789) costs about as much as missing
two of three correct ones (0.714)**. And for the 5.6% of businesses that are singletons the rule is
brutal — predicting an empty list scores **1.0**, predicting *anything* scores **0.0**.

The final score is a **macro average**: F0.5 is computed per business, then averaged over all
businesses. A tiny business counts as much as a famous one.

### What we concluded from the metric

Three consequences shaped everything downstream:

1. **Silence is often the best answer.** The decoder (Step 10) is built to output nothing unless it
   is confident.
2. **Caution is not free, but being wrong is expensive.** We tune every threshold against F0.5
   itself — not against accuracy, not against AUC.
3. **Sibling traps are the main enemy**, because they are exactly what converts a cheap miss into an
   expensive false merge.

The metric lives in [src/metric.py](code/business_entity_resolution/src/metric.py) — 25 lines, with
a self-test against the worked example from the challenge README. Implementing the official metric
yourself, early, and testing it, is a free way to eliminate a whole class of silent mistakes.

---

## 4. The pipeline at a glance

```
          raw .tsv files (12M records)
                    │
   ① NORMALIZE      │  clean names/addresses; translate 9 Indian scripts with a dictionary
                    │  MINED FROM THE TRAINING ANSWERS; parse numbers ("1577/15" → 1577, 15)
                    ▼
   ② BLOCKING       │  9 kinds of cheap shared "keys" propose ~300 candidates per business,
                    │  then two learned rankers cut that to 15 per source (28.6 total)
                    ▼
   ③ META-BLOCKING  │  a third learned model cuts 28.6 → 5.8 candidates   ──► candidate_pairs.tsv
                    ▼
   ④ FEATURES       │  73 numbers per surviving pair: similarities, number agreement,
                    │  IDF-weighted overlap, and "rival" context
                    ▼
   ⑤ MATCHERS       │  LightGBM #1 (trained on all candidates) and LightGBM #2 (trained on the
                    │  pruned ones it will actually see)
                    ▼
   ⑥ CROSS-ENCODER  │  a 140M-parameter multilingual transformer re-reads only the ~13% of pairs
                    │  the matchers are unsure about
                    ▼
   ⑦ STACKER        │  a model over all the above probabilities **plus the probabilities of
                    │  competing pairs** (collective classification)
                    ▼
   ⑧ DECODING       │  one owner per record; keep the prefix that maximizes *expected* F0.5;
                    │  per-country thresholds; stay silent when unsure
                    ▼
          matching_results.tsv  →  leaderboard
```

The shape to notice is a **funnel**. Cheap filters first, expensive models last. Stage ① touches
12M records, stage ② touches billions of key matches, stage ⑥ touches 1.8M pairs — because a
transformer costs on the order of 100,000× more per pair than a string comparison. This
retrieve-then-rank architecture is standard in search and recommendation systems, and it is the
only way this problem fits on a laptop.

---

## Step 1 — Reading the files without corrupting them

**What.** Load seven tab-separated files of ~500 MB each.

**Why it deserves a section.** The most common way to lose points in a data competition is not a bad
model — it is silently mangling the input. Three traps hit this dataset:

- A business named `"The" Bakery` contains a quote character. Default CSV parsers treat `"` as a
  quoting character and will swallow the rest of the line. Fix: `quote_char=None`.
- A country code can be `NA` (Namibia). Default parsers turn the string `"NA"` into *missing data*.
  Fix: never pass a null-values list, and keep every column as a string.
- Automatic type detection may read an ID column as a number, dropping leading zeros. Fix:
  `infer_schema=False` — everything is text until we decide otherwise.

**How here.** [src/data_io.py](code/business_entity_resolution/src/data_io.py) does all reading in
one place, and then does something small and valuable: it **counts the lines in the file itself and
asserts that the parser produced the same number of rows**.

```python
expected = _count_rows(path)
assert df.height == expected, f"{path}: parsed {df.height} rows, file has {expected}"
```

If a parsing bug ever appears, the run dies immediately instead of producing a slightly wrong answer
three hours later.

**Takeaway:** assert your assumptions at the boundary of the system. It is cheap, and it converts
silent corruption into a loud crash.

---

## Step 2 — Normalization: making two spellings look alike

**What.** Rewrite every name and address into a canonical form, so that two spellings of the same
thing become the same string — or at least a much closer one.

**Why.** Everything downstream — key lookups, string similarity, token overlap — compares text. If
one source writes `Private Limited` and another `Pvt. Ltd.`, every later stage has to fight the same
battle. Normalizing once, in one place, pays for itself everywhere.

**How here** ([src/normalize.py](code/business_entity_resolution/src/normalize.py), with the
hand-written rule lists in [src/rules.py](code/business_entity_resolution/src/rules.py)):

| Operation | Example |
|---|---|
| Lowercase, strip accents | `Léarning` → `learning` |
| Drop injected junk, IDs, brackets, honorifics | `Smt [Services] (ID: 13882)` → *gone* |
| Canonicalize legal forms | `Private Limited`, `Pvt-Ltd`, `PVT. LTD.` → `pvt ltd` |
| Split the name into `nm` (all tokens) and `core` (no legal forms, no honorifics) | `shyam consulting pvt ltd` → core `shyam consulting` |
| Keep a space-free name for website-style names | `shyamconsulting.com` → `shyamconsulting` |
| Expand address abbreviations, map state names to codes | `Road` → `rd`, `Maharashtra` → `mh` |
| Drop unit words that only add noise when *finding* candidates | `Shop No.`, `Flat`, `Plot` |
| Parse the number structure | `1577/15` → numbers `[1577, 15]`, structure `1577-15`, main `1577` |

Each record ends up with eleven derived columns, cached as Parquet files in `work/norm/`. **Parquet**
is a compressed columnar format: reading only the two columns you need out of a 5M-row table takes
under a second, whereas re-parsing the TSV takes minutes. Caching normalized data is the cheapest
speed-up in any pipeline like this.

### The interesting part: a transliteration dictionary mined from the answers

23% of India's S2 matches have names in Indian scripts. The obvious approach is a rule-based
transliterator (Devanagari → Latin, letter by letter). That turns `कंसल्टिंग` into `kansalTinga`
while the S1 record says `Consulting` — close enough to look reasonable, far enough to break exact
key matching.

Reading the data gave a better idea: these names are **word-by-word renderings of the English
name**, and there are only ~1,500 distinct such words in the entire dataset. So we *learn* the
dictionary from the training labels:

```
For every training pair (S1 record, matched S2/S3 record) whose match has an Indian-script name:
    if the two names have the same number of tokens:
        align them position by position and count every (Indian token → English token) pair
Keep a translation when it was seen at least twice and is the majority reading (≥50%).
```

That is [translit.mine()](code/business_entity_resolution/src/translit.py). The resulting dictionary
covers **96.4%** of the Indian-script words that appear in **test**; anything unseen falls back to
rule-based transliteration. Addresses get the same treatment at segment level, because Indian-script
address segments are almost always state names.

**Why this is worth studying:** it is a textbook example of using labels as a *data source*, not
only as a training target — and of staying inside the rules. No external dictionary, no translation
API, nothing but the provided training answers.

---

## Step 3 — Blocking: from 17 trillion pairs to 28 per business

**What.** **Blocking** proposes a shortlist of plausible candidates for each entity, so the
expensive model never sees the other 17 trillion pairs.

**Why it is the most important stage.** Blocking sets the **recall ceiling**. If the true match is
not in the shortlist, no downstream model — however clever — can recover it. A pair the blocker
misses is lost forever. We therefore spent as much effort here as on the classifier, and we measure
blocking recall after every change.

### 3.1 Multi-key blocking

A **blocking key** is a cheap string that two records must share to be worth comparing. Take `house
number + rare address word`: build that key for every record, group by key, and every group becomes
a set of candidate pairs. This is one database join, not 17 trillion comparisons.

One key is never enough, because any single key dies on some noise pattern (a typo in the name, a
missing house number). So [src/blocking.py](code/business_entity_resolution/src/blocking.py) uses
**nine kinds**, and a pair becomes a candidate if it shares *any* of them:

| Key | What it is | The noise it survives |
|---|---|---|
| `nw` | address number + one of the 4 rarest address words | the name is completely different (trade names) |
| `np` | the 2 rarest core-name words, as a pair | the address is missing or rewritten |
| `nt` | a single very rare name word (document frequency ≤ 30) | almost everything else is damaged |
| `cc` | the whole name with spaces removed | website-style names, spacing differences |
| `ap` | the 2 rarest address words, as a pair | numbers are missing or wrong |
| `nn` | address number + a rare name word | both sides partly damaged |
| `hw` | the full house-number structure (`2-1`) + a rare address word | unit-number detail matters |
| `nr` | a moderately rare name word — only for records with **no number** in the address | the 4.4% of records with no usable address |
| `ca` | space-free name + one of the 2 rarest address words (**added in v10**) | a common name shared by many businesses in one city |

Four details that each earned measurable points:

- **"Rare" requires document frequency ≥ 2.** A word appearing exactly once is almost always a typo
  — the *other* record cannot possibly share it, so using it as a key wastes the slot. This change
  alone lifted pair recall from 93.3% to 95.7%.
- **Every key block is capped** (for example at 200 records). If 50,000 records share a key, that
  key is uninformative *and* would generate 2.5 billion pairs; skip it.
- **Blocking runs per country**, because true pairs always share a country: smaller joins, zero
  recall cost.
- **Word frequencies come from each split's own text.** Counting words in the test files is
  *unsupervised* — it uses no labels — which keeps us inside the fair-play rules.

The union of all key matches is ~300 candidates per business and contains **98.4%** of true pairs.
That is our ceiling. Now we must cut 300 down to something affordable without losing much of it.

### 3.2 Learned ranking: a cascade of two rankers

The obvious way to cut is a hand-made score ("more shared keys, and rarer keys, means better"). We
built that first — the `pre` column, a sum of `1/log2(1 + block size)` over the shared keys. Keeping
its top 15 per source gave **95.8%** recall.

Then we replaced the hand-made formula with a **learned ranker**: a LightGBM model trained on
sampled training entities, whose target is "is this pair a true match?" and whose inputs are only
cheap blocking evidence (the pre-score, how many keys matched, *which kinds* matched, the pair's
rank within the entity's list). Learning the weights beats guessing them.

Because computing string similarities for 300 candidates per entity is itself expensive, it runs as
a **cascade** — the standard cheap-filter-then-expensive-filter pattern:

```
~300 key matches per business
      │  stage 0:  LightGBM on key evidence only (no string work at all)
      ▼  keep the best 100 per source
   ~140 per business
      │  stage 1:  LightGBM on key evidence + 4 cheap string similarities
      ▼  keep the best 15 per source
   28.6 candidates per business   → pair recall 98.0%
```

**The v10 lesson, and a nice piece of debugging.** In v7 the stage-0 cut kept 50 per source and
final recall was 97.1%. Rather than guess where the loss was, we printed recall at *every* step
(`blocking.py`'s `diag=True` mode):

| Step | Pair recall |
|---|---|
| Union of key matches | 98.4% |
| After stage 0's cut at 50 (v7) | 96.7% |
| Final top 15 per source (v7) | 97.1% |
| After stage 0's cut at 100 (v10) | 98.1% |
| **Final top 15 per source (v10)** | **98.0%** |

The loss was in the **first** cut, not the last one. Widening stage 0 recovered ~0.9 points of
recall — and, crucially, **the final candidate count per entity did not change** (still 28.6), so
nothing downstream got slower. Only the blocking stage itself pays.

**Takeaway:** instrument every stage. Had we tuned the final cut (the intuitive suspect) we would
have paid for extra candidates through the entire rest of the pipeline and gained less.

---

## Step 4 — Meta-blocking: a model that shortens the shortlist

**What.** **Meta-blocking** means pruning the candidate set using the structure of the candidate set
itself. Here it is *supervised* meta-blocking: a small LightGBM ranker
([src/metablock.py](code/business_entity_resolution/src/metablock.py)) scores each surviving
candidate and keeps it only if its probability is ≥ 0.01 — plus each entity's single best candidate,
always, so no entity is left with nothing.

**Why.** 28.6 candidates per business × 1.7M businesses = 49M pairs, and every later stage (73
features, two matchers, a transformer, a stacker) pays per pair. A threshold-based filter is also
**adaptive**: an unambiguous business keeps 1-2 candidates, a business in a crowded building keeps
10. A fixed "top 5" cannot do that.

**Numbers:** 28.6 → **5.78** candidates per entity, pair recall 0.9803 → 0.9788. We paid 0.15% of
recall to delete 80% of the work. On held-out entities the final F0.5 did not move at all
(0.9712 → 0.9712 when first measured).

A deliberate subtlety: the ranker's *context* inputs (a pair's rank within its entity, how many
entities compete for the same record) are computed **before** pruning. Computed after, each pair's
features would depend on which other pairs happened to survive, and the training-time and test-time
distributions would quietly diverge.

The kept set is exactly what we submit as `candidate_pairs.tsv`: **12,040,830 pairs, 6.95 per
business** — down from ~1.7 × 10¹³ possible ones, a reduction by a factor of roughly 1.4 million, at
the cost of 2% of the achievable matches.

---

## Step 5 — Features: turning two records into numbers

**What.** A **feature** is one number describing a pair. A model never sees text; it sees a row of
numbers. Feature engineering is deciding *which* numbers.

[src/features.py](code/business_entity_resolution/src/features.py) produces 66 per pair, and
[src/featidf.py](code/business_entity_resolution/src/featidf.py) adds 7 more. They fall into four
families.

### 5.1 String similarity — how alike is the text?

We use [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) for several *different* notions of
similarity, because each fails differently:

- **ratio** — overall edit similarity. Catches typos: `synagouge` vs `synagogue`.
- **token_set_ratio** — compares the *sets* of words, ignoring order and duplicates. Catches
  reordering: `laxminarayan chs eksar` vs `eksar laxminarayan chs`.
- **token_sort_ratio** — sorts the words, then compares. A middle ground.
- **partial_ratio** — is one string roughly contained in the other? Catches `Great Charities` vs
  `Great Charities Westgate`.
- **Jaro-Winkler** — rewards agreement at the *start* of the string, where business names carry most
  of their identity.

Giving a tree model five overlapping similarity measures is not redundancy: the model learns which
one to trust in which situation (for instance, trust `partial_ratio` only when the address also
agrees).

### 5.2 Set and number agreement — what do they literally share?

Computed with set operations over token lists: shared and extra name words, **Jaccard** overlap
(`shared / union`), legal-form agreement, and the number block that decides the sibling traps — is
the main house number equal, is the full number structure equal (`1577-15` vs `1577-8`), how many
numbers do the addresses share, how far apart are the main numbers. Plus bookkeeping flags: is the
candidate's address empty, was the name in an Indian script, does it look like a website.

### 5.3 IDF-weighted overlap — which shared words actually matter? (added in v10)

A shared word is not worth a fixed amount. Two records sharing `beloumbra` is near-proof; two
records sharing `traders` or `road` means nothing. Token-set ratios cannot express that — they count
every word the same.

**IDF** (inverse document frequency) is the classic fix: a word's weight is `log(N / df(t))`, where
`df(t)` is how many records contain it. Rare word, high weight. Then:

```
idf_name_jac   = Σ idf(words both sides share) / Σ idf(all words either side has)
idf_name_cov1  = Σ idf(shared) / Σ idf(S1's words)          "how much of us is in them"
idf_name_cov2  = Σ idf(shared) / Σ idf(candidate's words)    "how much of them is in us"
idf_name_miss  = idf of the most distinctive word one side has and the other lacks
```

…and the same three for addresses. That last one is the sibling-trap detector: *Agencies Limited* is
a cheap difference, a rare distinctive word is an expensive one.

A detail worth copying: when a side has no usable tokens these features are left **null**, not zero.
Zero means "no overlap"; null means "unknown". LightGBM handles nulls natively and learns separate
behaviour for them. Encoding "missing" as 0 is a very common beginner bug.

### 5.4 Context and rival features — the highest-signal group

If you remember one idea from this section, make it this one.

A pair is not an island. Ask not only *"do these two records look alike?"* but **"does anybody else
look better?"**

```
S1: "Baba Balaji Ventures, 1577/8"
      ├── candidate A: "Baba Balaji Ventures, 1577/8"           ← the pair we are scoring
      └── candidate B: "Baba Balaji Ventures Agcy, 1577/15"

A viewed alone:      name 100%, address 100%          → looks like a match
A viewed in context: also rank 1 for this entity, and no other S1 business wants it
                     more strongly                     → still a match
The mirror case:     a candidate that looks good to us but looks *better* to some other
                     S1 business is probably that business's record, not ours.
```

So, from the blocking pre-score only, we compute:

- **entity view** — this candidate's rank among the entity's candidates, its score relative to the
  entity's best, how many candidates there are;
- **rival view** — how many *other* S1 businesses compete for this same record, and this pair's rank
  among them;
- **consistency** — similarity between this candidate and the entity's *best other* candidate. If
  our top candidate is on Eksar Road and this one is too, they corroborate each other.

In the first model the rival features were the **top two features by gain** — more informative than
any string similarity. They are precisely the signal the sibling traps are designed to defeat, and
they are why the project later adds an entire collective stage (Step 9) built on the same idea.

One rule throughout: **no country indicator is ever a feature.** France exists only in test, so a
model allowed to key on country would learn US/India-specific behaviour and have nothing to say
about France. Deliberately withholding a feature to force generalization is a real technique, and
this dataset makes it mandatory.

---

## Step 6 — The matcher: what LightGBM actually does

**What.** A model that takes the 73 numbers of a pair and outputs the probability that the pair is a
match.

### Gradient-boosted trees in sixty seconds

A **decision tree** is a flowchart of yes/no questions: *is the address token-set ratio above 82?* →
*is the main house number equal?* → … → a prediction at the leaf. One tree is weak and crude.

**Gradient boosting** builds many trees in sequence, where each new tree is trained to fix the
errors *still remaining* after all the previous ones. The prediction is the sum of all the trees'
outputs, squeezed into [0,1]. **LightGBM** is a fast, memory-efficient implementation — and MIT
licensed, which the challenge rules require.

### Why trees and not a neural network?

For tables of hand-made features, gradient-boosted trees are usually the strongest and almost always
the most practical choice:

| | Gradient-boosted trees | Neural network on the same table |
|---|---|---|
| Mixed scales (0-100 ratios, log-counts, flags) | handled natively | needs careful scaling |
| Missing values | handled natively | needs imputation |
| Threshold interactions ("ratio > 82 *and* number equal") | what trees *are* | learnable, but slower |
| 12M rows on a laptop CPU | minutes to an hour | GPU, hours |
| Feature importances | free | harder |

We do use a neural model later (Step 8), but only where *text understanding* rather than tabular
reasoning is the bottleneck, and only on the 13% of pairs that need it. **Match the model class to
the shape of the problem** rather than to what is fashionable.

### The settings, and why

From [src/train.py](code/business_entity_resolution/src/train.py):

```python
PARAMS = dict(objective="binary",       # predict a probability, optimize log-loss
              learning_rate=0.1,        # each tree contributes 10% of its correction: slower, safer
              num_leaves=127,           # tree capacity: higher = more complex interactions
              min_data_in_leaf=200,     # a leaf must cover >=200 pairs: blocks memorizing single pairs
              feature_fraction=0.8,     # each tree sees 80% of features )  decorrelates the trees,
              bagging_fraction=0.8,     # and 80% of the rows            )  i.e. less overfitting
              lambda_l2=1.0,            # penalizes extreme leaf values
              seed=2026)                # reproducibility
```

**Overfitting** is a model learning the quirks of the training rows instead of the general pattern;
it looks brilliant in training and fails on new data. `min_data_in_leaf`, the two fractions and
`lambda_l2` are all defences against it. We also use **early stopping**: train up to 1000 rounds but
stop when the score on held-out data stops improving (it stopped near 988), so the number of trees
is chosen by evidence rather than by us.

The first matcher trains on the candidates of 400k entities (11.4M pairs). Held-out macro F0.5:
**0.9786**.

---

## Step 7 — A second matcher, trained on the right distribution

**What.** A second LightGBM — same features — trained on a *different sample* of pairs:
the ~5.8 meta-blocked candidates per entity rather than the ~28 raw blocking candidates.
[src/train_pruned.py](code/business_entity_resolution/src/train_pruned.py).

**Why.** The first matcher was trained on all 28 candidates per entity, but at prediction time it
only ever sees the 5.8 that survived meta-blocking — and those are *systematically harder*, because
the obvious non-matches were already deleted. A mismatch between the data a model is trained on and
the data it meets in production is called **train/serve skew**, and it is one of the most common
reasons a model underperforms once deployed.

So: train on exactly the distribution you will be scored on. Alone, the second matcher scores 0.9756
weighted versus 0.9746 for the first. But the real win is that we now have **two opinions** on every
pair, from models with different blind spots, and the stacker can use both as features — a form of
**ensembling**. It also covers all 2.2M training entities instead of a 400k sample, with ~1,200
trees chosen by early stopping.

---

## Step 8 — The cross-encoder: a small transformer for the hard cases

**What.** A pretrained multilingual language model that reads both records as one piece of text and
outputs a match probability:

```
input:  "shyam consulting pvt ltd | 1-D-206 eksar road borivali west mumbai mh"
        [SEP]
        "श्याम कंसल्टिंग प्रा. लि. | एक्सार रोड बोरीवली मुंबई महाराष्ट्र"
output: 0.94
```

The model is `jhu-clsp/mmBERT-small` — 140M parameters, MIT licensed (the challenge allows up to
8B). See [src/kaggle/xenc_kaggle.py](code/business_entity_resolution/src/kaggle/xenc_kaggle.py).

### Bi-encoder vs cross-encoder — a distinction worth learning

- A **bi-encoder** turns each record into a vector *independently*, then compares vectors (cosine
  similarity). Cheap — you can precompute 10M vectors and search them fast — but the two records
  never "see" each other, so it captures only broad semantic similarity. We tried one
  ([src/embed.py](code/business_entity_resolution/src/embed.py), Model2Vec): +0.0002 F0.5. Left off.
- A **cross-encoder** reads both records *together*, so every word of one can attend to every word of
  the other. Much stronger on fine distinctions — exactly the ones we need (`1577/8` vs `1577/15`) —
  but it costs a full model run *per pair*, so it cannot be used at retrieval scale.

**Why it helps here.** Our features are hand-made similarity numbers. A transliterated Hindi name, a
trade name, an ALL-CAPS French address: these need language understanding that `token_set_ratio`
does not have. mmBERT is pretrained multilingually, so it has already seen Devanagari, Tamil and
French before we show it anything.

### The two decisions that make it affordable and honest

**1. Only score the uncertain band.** We run it on pairs where the second matcher said
`0.02 < p < 0.98` — about 13% of pairs (1.60M train, 1.81M test). Where LightGBM already says 0.001
or 0.999, a transformer adds nothing. That is the funnel principle again, applied to the most
expensive component.

**2. Two-fold out-of-fold training.** Training entities are split into two halves. The model trained
on half A scores half B and vice versa, so **no pair is ever scored by a model that trained on its
own entity's label**. Test pairs get the average of both models. Without this discipline the
cross-encoder's training scores would be inflated, the stacker would learn to trust them far too
much, and the leaderboard would punish us — the failure mode called **leakage** (Step 11).

### Did it work?

Measured by **AUC** — the probability that a random true pair scores above a random false pair
(0.5 is random, 1.0 perfect) — on exactly the same uncertain pairs:

| Scorer | Out-of-fold AUC on the uncertain band |
|---|---|
| LightGBM matcher | 0.931 |
| multilingual-e5-small (118M) | 0.903 *(pilot, 20k pairs)* |
| **mmBERT-small (140M)** | **0.955** |

A pilot on 20k pairs picked the architecture before committing GPU hours to the full run — cheap
insurance. The full job (fine-tune both folds, score 3.4M pairs) took about an hour on a Kaggle T4
GPU; everything else in this project is CPU-only. Note what Kaggle was used for: **compute**, not
data. The only thing downloaded was the pretrained open-weights model, which the rules allow.

Adding the cross-encoder moved the internal score from 0.9834 to **0.9873** — the largest single
gain of the v10 round.

---

## Step 9 — The collective stacker: decisions that look at each other

**What.** A final LightGBM that re-scores every pair using not only that pair's own probabilities but
the probabilities of its **neighbouring pairs**.
[src/stack.py](code/business_entity_resolution/src/stack.py).

**Why.** Two ideas combine here.

**Stacking** (stacked generalization) means feeding several models' predictions into a further model
instead of averaging them. The second-level model can learn *when* to trust each first-level model —
"trust the cross-encoder when the address is empty", for example.

**Collective classification** means the decisions are not independent. In entity resolution they
genuinely are not: if record X is almost certainly business A's, then X is almost certainly *not*
business B's. A pairwise classifier cannot express that, because it sees one pair at a time. So we
hand the model the context explicitly — 30 features:

| Group | Features | The question it answers |
|---|---|---|
| Own probabilities | first matcher's `p`, second matcher's `p` | What do our two matchers think? |
| Entity view | rank of `p` in the entity's list, entity's best other `p`, gap to it, number of confident candidates, sum of `p`, list size | Is this the entity's best option, or its fifth-best? |
| Record view | the best `p` any **other** entity gives this record, the gap to it, how many entities compete | Does someone else want this record more? |
| Transitivity | name/address similarity and same-house-number between this candidate and the entity's two most probable other candidates, plus their `p` | Does this candidate agree with our other matches? |
| Cross-encoder | its score, its rank within the entity, the gap, the margin over rivals | What does the transformer think, in context? |

The transitivity block deserves a note: matching should be *consistent*. If an entity's confident
matches are all at `Eksar Road, Borivali`, a candidate at the same address gains credibility and one
40 km away loses it. That is the reasoning a human would use, written as four numbers.

**Iteration.** The stacker can run twice (`--iterate`): after round 1, the neighbour features are
recomputed from the *round-1* probabilities and a second stacker is trained on those. The intuition
is message passing — each round, information about rivals and co-matches travels one step further
through the candidate graph. In v7 this earned +0.0001; in v10 round 2 was *not* better than round 1
and was therefore dropped (`"round2": false` in the report). A gate decides, not a hunch.

**The gate.** `stack.py` refuses to write a submission unless the weighted out-of-fold F0.5 improves
by at least `MIN_GAIN = 0.001`. Across 20+ experiments, this is what kept the pipeline from
accumulating changes that merely looked good.

Result: the stacker lifted the pruned out-of-fold score from 0.9804 to **0.9873** (+0.0069).

---

## Step 10 — Decoding: from probabilities to an answer sheet

**What.** We now have a probability for all 12M candidate pairs. The submission needs a *list* per
business. Turning scores into a discrete answer is called **decoding**, and under F0.5 it is worth
real points. [src/decode.py](code/business_entity_resolution/src/decode.py) — 56 lines that matter.

### Rule 1: one owner per record

We verified on 2.2M training entities that no S2/S3 record ever belongs to two S1 businesses. So
each record is awarded only to the business that scores it highest; every other claim is dropped.
A **hard constraint** read off the data — free precision.

### Rule 2: keep the prefix that maximizes *expected* F0.5

The naive rule is "predict every pair with p > 0.5". That is wrong twice over: 0.5 is arbitrary, and
the metric is not symmetric. The real question is *which list maximizes the F0.5 we expect to get?*

Sort an entity's candidates by probability and consider keeping the top `k`:

```
expected true positives of the top k   ≈  p₁ + p₂ + … + p_k
expected number of true matches        ≈  (sum of ALL candidate probabilities) + miss_mass
expected precision                     =  (p₁+…+p_k) / k
expected recall                        =  (p₁+…+p_k) / expected number of true matches
                                            ↓
                              F0.5 for that k, computed directly
Choose the k with the highest value.
```

Worked example — candidates at `p = [0.95, 0.80, 0.45]`:

| k | Expected TP | Precision | Recall (expected true ≈ 2.25) | F0.5 |
|---|---|---|---|---|
| 1 | 0.95 | 0.95 | 0.42 | 0.754 |
| **2** | **1.75** | **0.88** | **0.78** | **0.855** |
| 3 | 2.20 | 0.73 | 0.98 | 0.773 |

The procedure stops at 2 by itself. No hand-tuned cutoff decided that — the metric did. Adding the
third candidate at p = 0.45 would raise recall and *lower* the score, which is exactly the
precision-heavy behaviour F0.5 asks for.

`miss_mass` is a prior for matches blocking never proposed. Without it the model believes its
candidate list is the whole truth and becomes over-eager.

### Rule 3: a singleton gate

If an entity's *best* candidate scores below `t_single`, output **nothing**. With 5.6% singletons,
each costing a full 1.0 point when wrongly filled, this is the most valuable decoding condition —
the v10 ablation shows it is now the only one that still measurably matters (removing it costs
0.00016 in the US, 0.00011 in India).

### Thresholds per country, and the country we cannot tune

The three parameters (`t_single`, `t_match`, `miss_mass`) are tuned by grid search **on out-of-fold
predictions over the full training universe**, so that every competing entity is present exactly as
at test time. Final values:

| Country | `t_single` | `t_match` | `miss_mass` | How chosen |
|---|---|---|---|---|
| US | 0.60 | 0.20 | 0.30 | tuned out-of-fold |
| India | 0.60 | 0.30 | 0.34 | tuned out-of-fold |
| **France** | **0.65** | 0.30 | 0.34 | **cannot be tuned — no labels exist** |

France is 15% of the test set and appears nowhere in training. We deliberately give it the
*stricter* setting: under F0.5, when you are uncertain about an entire country, erring toward silence
is the cheaper mistake. A leave-one-country-out study (Step 11) later confirmed this guess sits
within 0.0001 of the best value the data can support.

---

## Step 11 — How we measured everything (and avoided fooling ourselves)

This is the most transferable part of the project. Models are easy; trustworthy numbers are hard.

### The three numbers, and which one is real

| Number | What it is | Trust |
|---|---|---|
| **Holdout** | train on 400k entities, score 150k never-seen entities | quick and cheap |
| **Out-of-fold (OOF)** | split entities into 2 folds; each fold scored by a model trained on the other, so every entity gets a prediction from a model that never saw it | our main instrument |
| **Leaderboard (LB)** | the organizer's score on the real test set | **the only real score** |

Our v7 pipeline scored **0.9790 out-of-fold** and **0.968 on the leaderboard**. That ~0.011 gap is
not a bug, and understanding it is the whole point: the test set contains France (unseen,
unvalidatable) and is ~23% denser in distractor records. **Never quote an internal number as "the
score."** It is an instrument for comparing your own versions, not a prediction of your rank.

### Why out-of-fold, and why folds are by *entity*

A naive 80/20 split of *pairs* would leak: two candidate pairs of the same business share features
(ranks, rival counts), so a pair in validation could be effectively described by its sibling in
training. **Leakage** is any path by which information about the answer reaches the model through a
side channel; it inflates validation scores and evaporates at test time. Splitting by **entity**
closes that path, and we reuse the *same* entity folds for the matchers, the cross-encoder and the
stacker, so no component is ever scored by a model that saw its label.

Out-of-fold scoring also lets us decode over the **full universe**, which matters because the
one-owner rule involves every business competing for a record — it can only be evaluated faithfully
when predictions exist for everyone.

### The evaluation harness

[src/evaluate.py](code/business_entity_resolution/src/evaluate.py) exists because a single number
tells you nothing about *where* you are losing. It reports six things.

**1. Loss decomposition by consequence.** `1 - F0.5` split into parts that sum **exactly** to the
total (verified to 1e-16 in the report), using a sequential counterfactual: remove the wrong pairs
first, then add back the missed ones.

| Where the v10 loss goes | US | India |
|---|---|---|
| True matches never proposed as candidates (blocking) | 0.0058 | 0.0079 |
| Candidates proposed but scored too low (model) | 0.0048 | 0.0044 |
| Wrong merges (precision) | 0.0011 | 0.0011 |
| **Ceiling** if our candidates were scored *perfectly* | **0.9942** | **0.9921** |

This table is the project's to-do list, and it is why v10 attacked blocking first. It also says
something sobering: even a flawless matcher on today's candidates tops out near 0.993.

**2. Calibration by stratum.** A probability is **calibrated** if, among the pairs it scores 0.8,
about 80% really match. The harness checks calibration separately for pairs with and without house
numbers, for names shared by 1 / 2 / 3-5 / 6+ businesses, and for empty addresses. This is how we
found that the matcher was mis-calibrated on records whose name is shared by several businesses — it
could not *count* how many businesses share a name. That produced
[src/featctx.py](code/business_entity_resolution/src/featctx.py) (counting belongs in code, not in a
model's intuition), though it earned only +0.00007 and was not adopted.

**3. Decoding ablations.** Remove one decoding condition at a time and re-score, so we know which
rules still pay instead of assuming they all do. Finding: by v10 the one-owner rule is worth only
~0.00001, because the stacker already sees the rival signals. Rules can become redundant as models
improve — worth re-checking rather than keeping on faith.

**4. Honest thresholds.** Thresholds tuned on fold 0 and scored on fold 1, and vice versa. If the
honest numbers match the tuned ones, the tuning was not overfitting. They did: US 0.9882 both ways,
India 0.9866 both ways.

**5. A paired cluster bootstrap for confidence intervals.** Is +0.008 real, or noise? The
**bootstrap** resamples entities with replacement 1,000 times and recomputes the difference, giving
a 95% interval. The twist: entities competing for the same records are *not* independent, so they
are resampled together in clusters (connected components of the candidate graph; when one component
swallows 20% of all entities, clusters fall back to same-name groups). Result:

```
v10 - v7 weighted ΔF0.5 = +0.0083,  95% CI [+0.0082, +0.0085]
US  +0.0050 [+0.0049, +0.0051]      India  +0.0111 [+0.0109, +0.0112]
```

An interval far from zero — a real improvement, not a lucky run. Reporting a difference without an
interval is how teams talk themselves into believing noise is progress.

**6. Leave-one-country-out (LOCO).** [src/loco.py](code/business_entity_resolution/src/loco.py)
simulates France: train the matcher on **one** country and score the **other**, which it has never
seen. Findings: an unseen country costs roughly **-0.048** F0.5 on India-like data and **-0.012** on
US-like data, and the hand-picked France thresholds are within 0.0001 of the best the data supports.
We cannot validate France; we *can* measure what facing an unseen country costs, and set its policy
from evidence rather than optimism.

---

## Results, stated honestly

### Leaderboard — the real score

| Version | What changed | Public LB F0.5 |
|---|---|---|
| v2 | blocking key fixes (typo-proof rarity, wider key set) | 0.963 |
| v4 | learned two-stage blocking cascade | 0.966 |
| v5 | + collective second stage (stacker) | 0.967 |
| v6 | + second matcher on the pruned distribution | 0.968 |
| **v7 — final submitted version, 27 Sep 2026** | + a second collective round | **0.968** |
| v10 | wider stage-0, `ca` key, IDF features, cross-encoder | *not scored — built after the deadline closed* |

### Internal out-of-fold — our instrument, **not** a leaderboard prediction

| Out-of-fold macro F0.5 | US | India | Weighted (test mix 0.45 / 0.55) |
|---|---|---|---|
| v7 (submitted) | 0.9832 | 0.9755 | 0.9790 |
| v10 without the cross-encoder | 0.9851 | 0.9820 | 0.9834 |
| **v10 complete** | **0.9882** | **0.9866** | **0.9873** |

v10's internal gain over v7 is +0.0083, with a 95% interval of [+0.0082, +0.0085], reproduced by
fold-honest tuning. What that would have been worth on the leaderboard is **unknown**: v7's own
0.9790 internal corresponded to 0.968 public, and v10 was finished after the deadline.

Where v10's remaining loss sits: candidate generation (0.0058 US / 0.0079 India) is now the biggest
bucket, scoring misses next (0.0048 / 0.0044), wrong merges smallest (0.0011 / 0.0011) — exactly the
profile F0.5 rewards.

---

## Things we tried that did not work

Negative results are data. Everything here was built, measured against the gate, and rejected.

| Idea | Result | Why we walked away |
|---|---|---|
| **Reverse retrieval** — also keep each record's best S1 candidates, not only each S1's best records | +0.01-0.02 recall points for +2.3 candidates per entity | Bad trade: every later stage pays per candidate |
| **Bi-encoder retrieval** (Model2Vec `potion-multilingual-128M` + FAISS) | blocking recall on empty-address records +5-7 pts; final F0.5 **+0.0002** | The recovered pairs were name-only and inherently ambiguous — the decoder refused them anyway |
| **Local CPU cross-encoder** ([src/xenc.py](code/business_entity_resolution/src/xenc.py)) | ~13 pairs/second on the laptop | ~1M pairs would take weeks; moved to a Kaggle GPU instead |
| **Name-count context features** ([src/featctx.py](code/business_entity_resolution/src/featctx.py)) | +0.00007 [+0.00002, +0.00012] | Below the gate, and train/test densities differ, so it carries shift risk |
| **Look-alike / unit-suffix features** ([src/featv2.py](code/business_entity_resolution/src/featv2.py)) | built and unit-tested; the evaluation was interrupted | We do not adopt on an unfinished measurement |
| **Isotonic calibration per country** | lowered the unseen-country score | Calibration fitted on seen countries does not transfer |
| **Bigger stacker** (1,200 trees) / **wider trees** (255 leaves) | +0.0001 each | Below the gate: more compute for nothing |
| **A second collective round, in v10** | not better than round 1 | Adopted in v7, rejected in v10 — a gain is not permanent, so re-measure it |

The pattern: **a gate declared in advance (+0.001 weighted OOF) plus an honest measurement of every
idea.** Roughly half of these felt promising. Feelings are not a gate.

---

## Engineering lessons

Things that cost us real hours, in case they save you some.

**Memory is a design constraint, not a detail.** The laptop has 31 GB. Holding all 31M feature rows
of one fold at once (~10 GB) crashed the out-of-fold stage — `memory allocation of 126090228 bytes
failed`, visible in [work_v10_run.log](logs/work_v10_run.log). The fix in
[src/train_oof.py](code/business_entity_resolution/src/train_oof.py) is to score **one feature file
at a time**. More generally every heavy stage processes data in chunks and writes Parquet to
`work/`, and heavy stages run strictly **one after another** — running two in parallel once cost
about six hours of swapping.

**Cache every stage, keyed by whether its output exists.** A 9-hour pipeline that cannot resume is a
pipeline you will never finish debugging.

**Make experiments nameable.** `ER_TAG`, `--tag` and `--suffix` give every variant its own artifacts
(`oof_pruned_idf.parquet`, `oof_stack_v10x.parquet`), so two versions can be compared later instead
of overwriting each other. [src/evaluate.py](code/business_entity_resolution/src/evaluate.py) takes
an artifact *name* precisely because of this.

**Configure through the environment.** [src/config.py](code/business_entity_resolution/src/config.py)
reads `ER_DATA_DIR`, `ER_WORK_DIR`, `ER_K`, `ER_K0`, `ER_CA`… so the same code runs on the laptop, on
a cloud box and on Kaggle with no edits. `run_v10.sh` records the exact environment used — that file
*is* the reproducibility statement.

**Validate the submission with the organizer's own validator** before every upload. A rejected file
wastes one of ~3-5 daily uploads. [make_submission.py](make_submission.py) goes further: it refuses
to build the zip unless the `matching_results.tsv` inside is byte-identical (by MD5) to the file
actually uploaded, because the rules require that and it is trivially easy to get wrong.

**Windows specifics.** `set PYTHONIOENCODING=utf-8` before printing Devanagari or Tamil, or the
console raises `charmap codec can't encode character`. Git will also warn about LF→CRLF conversion
on shell scripts: harmless, but do not let it rewrite files you ship.

**Check that your inputs are the inputs you think they are.** One failed run
([work_v10_run2.log](logs/work_v10_run2.log)) came from a stale uploaded dataset: the GPU job scored
1,490,405 pairs while the band held 1,601,814. An explicit count check caught it and refused to
continue — the cheapest kind of safety net to write, and the one that prevents a wrong submission.

---

## How to run it yourself

You need the organizer's dataset, which is not in this repo (it is not ours to redistribute).

```bash
pip install -r code/business_entity_resolution/requirements.txt    # Python 3.13, CPU-only
cd code/business_entity_resolution/src
python metric.py          # sanity check: prints "metric OK"
python run_all.py         # the whole CPU pipeline, ~9 h on a 12-thread / 31 GB laptop
```

Paths default to the challenge layout and can be overridden with `ER_DATA_DIR`, `ER_WORK_DIR` and
`ER_OUT_DIR`. Every stage caches into `work/` and can be run on its own — see
[code/business_entity_resolution/README.md](code/business_entity_resolution/README.md) for the
per-stage commands, the GPU step and timings. The cross-encoder stage needs a GPU; without it the
pipeline still runs end to end and simply skips those features.

---

## File map

| File | What it does | Where it is explained |
|---|---|---|
| [src/config.py](code/business_entity_resolution/src/config.py) | paths and scale knobs, all from environment variables | Engineering |
| [src/data_io.py](code/business_entity_resolution/src/data_io.py) | safe TSV reading, submission writing | Step 1 |
| [src/rules.py](code/business_entity_resolution/src/rules.py) | hand-written abbreviation / legal-form / state lists | Step 2 |
| [src/translit.py](code/business_entity_resolution/src/translit.py) | mines the Indian-script dictionary from training matches | Step 2 |
| [src/normalize.py](code/business_entity_resolution/src/normalize.py) | name and address cleaning, parallel over all records | Step 2 |
| [src/blocking.py](code/business_entity_resolution/src/blocking.py) | nine-key candidate generation + the two learned rankers | Step 3 |
| [src/metablock.py](code/business_entity_resolution/src/metablock.py) | supervised meta-blocking | Step 4 |
| [src/features.py](code/business_entity_resolution/src/features.py) | the 66 pair features | Step 5 |
| [src/featidf.py](code/business_entity_resolution/src/featidf.py) | 7 IDF-weighted overlap features | Step 5 |
| [src/featv2.py](code/business_entity_resolution/src/featv2.py) | sibling-trap features (built, not adopted) | Not adopted |
| [src/featctx.py](code/business_entity_resolution/src/featctx.py) | name-count features (built, not adopted) | Not adopted |
| [src/train.py](code/business_entity_resolution/src/train.py) | first matcher + holdout threshold tuning | Step 6 |
| [src/train_oof.py](code/business_entity_resolution/src/train_oof.py) | full-universe out-of-fold scoring | Step 11 |
| [src/train_pruned.py](code/business_entity_resolution/src/train_pruned.py) | second matcher, on the pruned distribution | Step 7 |
| [src/xenc_export.py](code/business_entity_resolution/src/xenc_export.py) | exports the uncertain pairs for the GPU job | Step 8 |
| [src/kaggle/](code/business_entity_resolution/src/kaggle/) | the Kaggle GPU notebook that fine-tunes the cross-encoder | Step 8 |
| [src/xenc.py](code/business_entity_resolution/src/xenc.py) | earlier CPU cross-encoder (too slow, kept for reference) | Not adopted |
| [src/stack.py](code/business_entity_resolution/src/stack.py) | the collective stacker, test inference, submission files | Step 9 |
| [src/decode.py](code/business_entity_resolution/src/decode.py) | one-owner rule + expected-F0.5 decoding | Step 10 |
| [src/metric.py](code/business_entity_resolution/src/metric.py) | the official metric, with a self-test | Section 3 |
| [src/evaluate.py](code/business_entity_resolution/src/evaluate.py) | loss decomposition, calibration, ablations, bootstrap CIs | Step 11 |
| [src/loco.py](code/business_entity_resolution/src/loco.py) | leave-one-country-out study for France | Step 11 |
| [src/predict.py](code/business_entity_resolution/src/predict.py) | first-matcher test inference | Step 6 |
| [src/run_all.py](code/business_entity_resolution/src/run_all.py) | runs the whole pipeline in order | Step 4 overview |
| [make_submission.py](make_submission.py) | builds the official zip, with an MD5 identity check | Engineering |

---

## Glossary

**Ablation** — removing one component and re-measuring, to learn what it was worth.
**AUC** — the probability that a random positive scores above a random negative; 0.5 is random.
**Bi-encoder** — a model that encodes each text separately, then compares the vectors.
**Blocking** — cheaply proposing a shortlist of candidate pairs so the model never sees all pairs.
**Bootstrap** — resampling the data many times to put a confidence interval around a number.
**Calibration** — whether a predicted probability of 0.8 really means "right about 80% of the time".
**Candidate pair** — a (business, record) pair that the model will score.
**Cross-encoder** — a model that reads two texts *together* and scores their relationship.
**Decoding** — turning scores into the final discrete answer (which list to output).
**Document frequency (df)** — in how many records a word appears. Low df = rare = informative.
**Early stopping** — stop adding trees or epochs once held-out performance stops improving.
**Ensembling** — combining several models' predictions.
**Entity resolution** — deciding which records refer to the same real-world thing.
**F0.5 / F1** — combinations of precision and recall; F0.5 weights precision 2× recall.
**Feature** — one number describing an example, fed to a model.
**Gradient boosting** — building many small models in sequence, each fixing the previous errors.
**Ground truth** — the correct answers, provided for the training split only.
**Holdout** — data kept out of training and used to estimate performance.
**IDF** — `log(N / df(t))`: a weight that makes rare words count for more.
**Jaccard** — `|shared| / |union|` for two sets.
**Leakage** — information about the answer reaching the model through a side channel, which makes
validation scores lie.
**Macro average** — compute the metric per entity, then average; every entity counts equally.
**Meta-blocking** — pruning the candidate set using the structure of the candidate set.
**Out-of-fold (OOF)** — every example scored by a model that was trained without it.
**Overfitting** — learning the training data's quirks instead of the general pattern.
**Parquet** — a compressed columnar file format; fast when you need only a few columns.
**Precision / recall** — of what you predicted, how much was right / of what was right, how much you
predicted.
**Recall ceiling** — the maximum recall still reachable after an earlier stage threw things away.
**Singleton** — an entity with no matches; the correct output is an empty list.
**Stacking** — feeding models' predictions into another model as features.
**Threshold** — the cutoff that turns a probability into a decision.
**Train/serve skew** — the data a model is trained on differing from the data it meets in production.
**Transliteration** — rewriting text from one script into another (Devanagari → Latin).

---

*Written by team **gemma_gals** — Soumya Sharma (leader), Anjali Singh, Prisha Raj — BIT Mesra,
Patna Campus. Amazon ML Challenge 2026.*
