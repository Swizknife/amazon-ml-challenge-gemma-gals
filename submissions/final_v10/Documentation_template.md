# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** gemma_gals
**Team Members:** Soumya Sharma (leader), Anjali Singh, Prisha Raj (Birla Institute of Technology, Mesra, Patna Campus)
**Submission:** final package, pipeline version v10 (post-deadline revision of the 27 Sep submission, v7)

---

## 1. Executive Summary
A pipeline in which the matching model is LightGBM and a small fine-tuned multilingual transformer re-reads the uncertain pairs:
1. **Multilingual normalization**, including a cross-script dictionary learned from the training matches.
2. **Learned candidate generation**: multi-key blocking whose ~300 key matches per entity are ranked by a **two-stage LightGBM cascade** (stage-0 shortlist widened to 100 in v10), then **supervised meta-blocking** down to ~5.8 candidates per entity (6.95 in `candidate_pairs.tsv` for test). 97.9% of true training pairs survive.
3. A **first LightGBM matcher** (66 features) and a **second matcher trained on the trimmed candidates** it actually scores, with 7 additional **IDF-weighted overlap features**.
4. A **cross-encoder** (`jhu-clsp/mmBERT-small`, MIT, 140M parameters) fine-tuned on the provided training pairs and applied only to the ~13% of pairs the matcher is unsure about.
5. A **collective second-stage model** that re-scores every pair from both matchers' probabilities, the cross-encoder score, and the predictions for competing entities and for the entity's other likely matches.
6. **Constraint-aware decoding** that picks each entity's match list with the highest expected F0.5, under a data-proven "one owner per record" rule.

**Results.** Out-of-fold macro F0.5 over all 2.2M training entities (US and India), measured with the full-universe protocol described in section 5:

| Version | US | India | Weighted (test mix 0.45 / 0.55) |
|---|---|---|---|
| v7 (27 Sep submission) | 0.9832 | 0.9755 | 0.9790 |
| v10 without the cross-encoder | 0.9851 | 0.9820 | 0.9834 |
| **v10 (this package)** | **0.9882** | **0.9866** | **0.9873** |

The paired cluster-bootstrap 95% interval of the v10 - v7 difference is +0.0083 [+0.0082, +0.0085]. **These are internal validation numbers on training data. They overstate the leaderboard**: v7 scored 0.9790 internally and 0.968 on the public leaderboard, because France (15% of test) is unseen in training and the test set is denser. The public leaderboard history of the earlier versions is 0.963 (v2), 0.966 (v4), 0.967 (v5), 0.968 (v6 and v7). v10 was completed after the deadline; its leaderboard score is not known at the time of writing.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA findings)
- **Structure:** 2.2M S1 and 10.3M S2/S3 records in train. S1 has 0-11 matches (median 3), and 5.6% are singletons. No S2/S3 record belongs to two S1 entities, and every true pair shares its country.
- **Number noise:** only 73% of true pairs share the first house number, and 4.4% of matched records have an empty address.
- **Indian scripts:** 22.7% of India-S2 matches have names in one of 9 Indian scripts. They are word-by-word renderings of the English name (vocabulary of 1.5k tokens). A dictionary mined from training pairs covers 96.4% of these tokens in test.
- **Planted hard negatives:** "sibling" businesses in the same building with a different unit. 25.6% of unmatched records look like some S1, and 36.6% of singletons have such a look-alike (68.5% in India).
- **Train-to-test shift:** test has 23% more S2/S3 records per S1, so orphan records are about 39% of test vs 26% of train. France (15% of test) is not in training.
- **Where the error is (v7, measured with the evaluation harness).** After the second stage, 44% (US) and 58% (India) of the lost F0.5 came from true matches never proposed as candidates, 40% / 28% from candidates scored too low, and 17% / 14% from wrong merges. This ordered the v10 work: candidates first, then the scorer.

### 2.2 Solution Strategy
**Approach Type:** blocking + learned candidate ranking + supervised meta-blocking + gradient-boosted matchers + transformer re-scoring of uncertain pairs + stacked collective classification + constrained decoding.
**Core elements:**
- a data-mined cross-script dictionary;
- rival and consistency features against sibling traps;
- learned candidate ranking and meta-blocking for a very small candidate set;
- one-owner, expected-F0.5 decoding with per-country thresholds;
- an evaluation protocol that measures each stage separately (section 5).

---

## 3. Candidate Generation (Blocking)
**Normalization:** Unicode and accent folding; Indian-script dictionary with ITRANS fallback; canonical legal forms (US/India/France); junk, ids and honorifics removed; address abbreviations and state codes; structured numbers (`1577/15` -> 1577, 15).

**Stage 1: multi-key blocking** (polars joins per country and source; each key block is capped). Key kinds: address number + rare address word; rare name-token pair and very rare single name tokens; concatenated name; rare address-word pair; number + name token; full house-number structure + word; moderate name tokens for records without an address number; and (v10) **`ca`**, the concatenated core name plus one of the record's two rarest address words, for common names. "Rare" only counts tokens seen in at least 2 records, so typos never take key slots. Token frequencies come from each split's own text.

**Learned ranking cascade.** The union of key matches (~300 per entity) holds 98.4% of true training pairs. Two LightGBM rankers choose which to keep: **stage 0** scores each pair by its per-kind key evidence and keeps the best **100** per source (v10; v7 kept 50), **stage 1** adds four cheap string similarities and keeps the best **15** per source.

**What the v10 diagnostic found** (training pairs, measured per stage):

| Step | Pair recall |
|---|---|
| Union of key matches (with `ca`) | 98.4% |
| Stage-0 shortlist, top 50 (v7; measured on India only) | 96.7% |
| Stage-0 shortlist, top 100 (v10, both countries) | 98.1% |
| Final top 15 per source, v7 | 97.1% |
| **Final top 15 per source, v10** | **98.0%** |

The stage-0 cut, not the final cut, was losing the true pairs: widening it recovered 0.9 points, and only the blocking step gets slower, because the final kept count per entity is unchanged (28.6 per S1). Reverse retrieval (also keeping each record's best S1 candidates) was tested and rejected: +0.01-0.02 points for 2.3 extra candidates per S1.

**Stage 2: supervised meta-blocking.** A 300-tree LightGBM ranker keeps a candidate if its probability is >= 0.01 (plus each entity's best), using blocking-derived signals and four cheap similarities. On train: 28.6 -> 5.78 candidates per S1 with pair recall 0.9803 -> 0.9788.

**Candidate pairs generated (test):** 49.4M after stage 1 (28.5 per S1), **12,040,830 after meta-blocking (6.95 per S1)**, down from ~1.7 x 10^13 possible pairs. `candidate_pairs.tsv` is exactly the set the final model scores.

---

## 4. Matching Model
**Features** (no country indicator, so the model transfers to unseen France):
- **Name** (ratio, token-set, token-sort, partial and Jaro-Winkler on core names; concatenated-name containment; shared and extra token counts; rarest-token cross-presence; legal-form agreement).
- **Address** (token-set and ratio on the full and word-only address; word Jaccard; rarest-word agreement; house-number structure: main equal, full structure equal, number-set Jaccard, numeric distance).
- **Context** (rank and relative score within the entity's list; **rival signals**, the candidate's rank among all entities competing for it; consistency with the entity's best other candidate; key-kind flags).
- **IDF-weighted overlaps (v10, `featidf.py`)**: with idf(t) = log(N / df(t)) per country from each split's own text, the weighted Jaccard and the coverage of each side for core-name tokens and for address words, plus the idf of the rarest token one side has and the other lacks. A shared rare word is strong evidence and a shared common one is weak, which token-set ratios cannot express.

**First matcher:** LightGBM binary classifier (MIT) on 400k entities (11.4M pairs); holdout macro F0.5 0.9786. **Second matcher:** the same model class trained on exactly the meta-blocked pairs it scores (12.8M pairs, all 2.2M entities, 2-fold on entity folds).

**Cross-encoder (v10).** `jhu-clsp/mmBERT-small` (MIT, 140M parameters; the limit is 8B) reads "name | address" of both records as one text pair (max length 64) and outputs a match probability. It is applied to pairs with 0.02 < p < 0.98 on the second matcher's probability (1.60M training pairs, 1.81M test pairs). It is fine-tuned on **training labels only**, 2-fold on the same entity folds as the matchers: model f trains on fold f's uncertain pairs (up to 400k, learning rate 5e-5, fp16) and scores fold 1-f; the test score is the mean of the two fold models. No pair is scored by a model that saw its entity's label. A pilot picked the architecture on a 20k-pair sample: mmBERT-small AUC 0.936, multilingual-e5-small 0.903, LightGBM matcher 0.928. On the full uncertain band the cross-encoder reaches an out-of-fold AUC of **0.955 against 0.931 for the matcher**. Training and inference ran on a Kaggle GPU (about 1 hour); Kaggle was used only as compute, and the only external download was the pretrained open-weights model.

**Second stage: collective stacking.** A 400-tree LightGBM re-scores each pair from the first-stage probabilities of related pairs, trained 2-fold on out-of-fold scores. Features (30): entity view (rank of p, best other p, gap, count of confident candidates, sum of p, rank within source); record view (the best p any other entity gives the same record, and the gap); transitivity (name and address similarity and same house number between the candidate and the entity's two most probable other candidates, plus their p); both matchers' probabilities; and the cross-encoder score with its rank, gap within the entity and margin against rival entities.

**Decoding.** After one-owner assignment (each S2/S3 record goes to the S1 that gives it the highest probability), each entity keeps the probability-sorted prefix that maximizes expected F0.5. Three parameters per country (singleton gate, minimum probability, missed-match prior) were tuned on full-universe out-of-fold scores in v6 and are reused unchanged: US 0.6 / 0.2 / 0.3, India 0.6 / 0.3 / 0.34 (`src/thresholds_v6b.json`). France has no labels and uses the stricter 0.65 / 0.3 / 0.34.

---

## 5. Evaluation Protocol, Results & Error Analysis
**Protocol.** Every score is out-of-fold over all 2.2M training entities, with all competing entities present, exactly as at test time. `evaluate.py` decodes with the submission's decoder (verified against the challenge metric implementation to 1e-8) and reports: the loss split by consequence (candidates never proposed, candidates scored too low, wrong merges); calibration by stratum; the effect of removing each decoding condition; a perfect-matcher ceiling; thresholds tuned on one entity fold and scored on the other; and a **paired cluster bootstrap** (1,000 draws; entities sharing candidate records are resampled together; 19.7% of entities fall in one connected cluster, so clusters fall back to groups of identical normalized names).

| Out-of-fold macro F0.5 | v7 | v10 | v10 vs v7 (95% CI) |
|---|---|---|---|
| US | 0.9832 | 0.9882 | +0.0050 [+0.0049, +0.0051] |
| India | 0.9755 | 0.9866 | +0.0111 [+0.0109, +0.0112] |
| Weighted | 0.9790 | 0.9873 | +0.0083 [+0.0082, +0.0085] |

Thresholds tuned on one fold and scored on the other give the same values (US 0.9882, India 0.9866), so the result is not an artifact of tuning on the scoring data.

**Where the remaining loss is (v10, F0.5 points):**

| | US | India |
|---|---|---|
| True matches never proposed | 0.0058 | 0.0079 |
| Candidates scored too low | 0.0048 | 0.0044 |
| Wrong merges | 0.0011 | 0.0011 |
| Ceiling with perfect scoring of our candidates | 0.9942 | 0.9921 |

Candidate generation is now the largest remaining bucket, and the cross-encoder removed more than half of the wrong merges and a third of the scoring misses compared with v7.

**Decoding ablation (v10, F0.5 lost when one condition is removed).** The singleton gate is the only condition that still matters (US -0.00016, India -0.00011); removing the one-owner rule costs about 0.00001, because the stacker has already seen the rival signals; removing the minimum-probability condition or the missed-match prior changes nothing.

**Unseen country.** A leave-one-country-out experiment (train on one country, score the other) measured the cost of an unseen country at about -0.048 F0.5 (India-like data) and -0.012 (US-like data) relative to training on it directly, and showed the hand-set France thresholds are within 0.0001 of the best value the data supports. France itself cannot be validated; this is why the leaderboard is expected to sit below the internal numbers.

**Common errors (sibling traps the data plants deliberately):** same name plus an extra word or legal form in the same building but a different unit; same name and street with a house number that differs by one dropped or changed digit (true pairs contain the same noise, so these are inherently ambiguous); identical names with an empty candidate address when several businesses share the name.

---

## 6. Conclusion
Measuring where the error was, not assuming it, drove the v10 changes: the loss decomposition pointed at candidate generation, a per-stage recall diagnostic found that the stage-0 cut (not the final cut or reverse retrieval) was dropping true pairs, and the uncertain pairs were handed to a transformer that reads both records together. Together they raise the internal out-of-fold score from 0.9790 to 0.9873 with a tight interval. The remaining loss is mostly candidate generation, and France cannot be validated without labels.

---

## Appendix
### A. Fair play and licenses
- No external databases, APIs, geocoding or entity-lookup services. Only the provided data is used.
- Test text is used only for unsupervised statistics (token frequencies and IDF); no test labels, no pseudo-labels, no test-time adaptation.
- Models: LightGBM (MIT); `jhu-clsp/mmBERT-small` (MIT, 140M parameters). Libraries: polars, rapidfuzz, numpy, scipy, scikit-learn, indic-transliteration (MIT/BSD), torch (BSD-3), transformers (Apache-2.0).
- The cross-encoder is trained only on the provided training labels; its weights are produced by the code in `src/kaggle/`.

### B. Code artefacts
`code/business_entity_resolution/src/run_all.py` runs normalize -> blocking -> matching models -> out-of-fold -> meta-blocking -> predict -> features -> second matcher -> stack; `run_v10.sh` and `run_v10_finish.sh` contain the exact commands used for this submission, and `src/kaggle/README.md` describes the GPU stage. The README gives per-stage commands and timings.

### C. Experiments evaluated and not adopted
| Experiment | Outcome |
|---|---|
| Reverse retrieval of candidates | +0.01-0.02 points of recall for +2.3 candidates per S1: left off |
| Name-count context features (how many S1s share a name) | +0.00007 weighted F0.5 [+0.00002, +0.00012]: not worth the train/test density risk; not included |
| Look-alike (unit, house-number suffix) features | built and unit-tested; the full evaluation was interrupted and not repeated, so not included |
| Isotonic calibration per country | lowered the unseen-country score: not used |
| Model2Vec embedding retrieval (earlier) | +0.0002 F0.5: left off (`ER_USE_EMB=1`) |
| Local CPU cross-encoder (`xenc.py`) | about 13 pairs/s on a laptop CPU, too slow: replaced by the Kaggle GPU job |
