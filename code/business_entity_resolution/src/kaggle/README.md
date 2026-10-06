# Cross-encoder on Kaggle GPU (v10, step A2)

The LightGBM matcher is unsure about ~10–15% of the candidate pairs. A small multilingual transformer reads
both raw records of each of those pairs and its score becomes a stacker feature (`stack.py --xenc oof_xenc`).
Training needs a GPU, so it runs as a private Kaggle notebook.

**Prerequisites:**
- a phone-verified Kaggle account (Kaggle allows GPUs and notebook internet only then)
- `~/.kaggle/kaggle.json`
- `pip install kaggle`

1. **Export the uncertain pairs** (challenge data: the dataset stays private):

   ```bash
   cd src && python xenc_export.py --out <WORK_DIR>/kaggle_data   # add --tag _v10 for a new pipeline run
   ```

   The first time, put a `dataset-metadata.json` in that folder:
   `{"title": "er-xenc-band", "id": "<user>/er-xenc-band", "licenses": [{"name": "other"}]}`.

   Then run:

   ```bash
   kaggle datasets create -p <WORK_DIR>/kaggle_data          # private by default; later: kaggle datasets version -p ... -m "msg"
   ```

2. **Run the notebook** (`kernel-metadata.json` here points at that dataset; GPU and internet on, private):

   ```bash
   kaggle kernels push -p src/kaggle
   kaggle kernels status <user>/er-xenc-v10                  # until "complete"
   kaggle kernels output <user>/er-xenc-v10 -p <WORK_DIR>    # oof_xenc.parquet, scored_test_xenc.parquet, report.json
   ```

   - **The default run:**
     - pilots `jhu-clsp/mmBERT-small` against `intfloat/multilingual-e5-small` (both MIT) on 100k pairs
     - keeps the higher out-of-fold AUC
     - trains one model per entity fold
   - **Folds are never mixed:** a pair is scored only by the model that did not train on its entity's fold.
   - **To score a new band with the saved fold models:**
     - add this kernel's output as a `kernel_sources` input
     - set `XENC_SCORE_ONLY=1`

3. **Stack:**

   ```bash
   python stack.py --second oof_pruned --primary second --xenc oof_xenc --save-oof ...
   ```

   Then gate it with `evaluate.py --vs <current best>`.
