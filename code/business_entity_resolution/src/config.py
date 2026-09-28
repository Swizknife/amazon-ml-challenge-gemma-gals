"""Paths and global constants. Override paths with ER_DATA_DIR / ER_WORK_DIR / ER_OUT_DIR."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get(
    "ER_DATA_DIR", ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", ROOT / "output"))

SEED = 2026
N_WORKERS = max(1, (os.cpu_count() or 4) - 2)

# ER_USE_EMB=1 adds the Model2Vec retrieval pairs and cosine features (see embed.py);
# ER_TAG suffixes model / threshold / prediction artifacts so variants can be compared.
USE_EMB = os.environ.get("ER_USE_EMB", "0") == "1"
TAG = os.environ.get("ER_TAG", "")

# Scale knobs (defaults fit a 31 GB laptop; a 256 GB machine can raise them):
K_PER_SOURCE = int(os.environ.get("ER_K", "15"))         # blocking candidates kept per S1 and source
K0 = int(os.environ.get("ER_K0", "50"))                   # stage-0 shortlist re-ranked by stage 1
CAP_MULT = float(os.environ.get("ER_CAP_MULT", "1"))      # multiplier on blocking key-block caps
N_TRAIN = int(os.environ.get("ER_N_TRAIN", "400000"))     # S1 entities used to fit the matcher
OOF_FIT = int(os.environ.get("ER_OOF_FIT", "400000"))     # S1 entities per out-of-fold model


def artifact(name, ext):
    return WORK_DIR / f"{name}{TAG}.{ext}"


def src_path(split, k):
    """Path of a source file, e.g. src_path("train", 2) -> .../train/train_source2.tsv."""
    return DATA_DIR / split / f"{split}_source{k}.tsv"


def norm_path(split, k):
    return WORK_DIR / "norm" / f"{split}_s{k}.parquet"
