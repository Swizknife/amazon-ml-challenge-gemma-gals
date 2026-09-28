"""Multilingual static embeddings with Model2Vec (MIT): minishlab/potion-multilingual-128M,
a 128M-parameter static distillation of BAAI/bge-m3 covering 101 languages.

Every record gets a 64-d name vector and a 64-d address vector (the model's leading
PCA dimensions, L2-normalized, stored as float16). They power:
  * fuzzy name retrieval (blocking kind `em`): each Source-2/3 record whose address
    has no number -- so it can only be found by its name -- is linked to its 5
    nearest Source-1 names in the same country (FAISS IVF, inner product);
  * the cosine features e_name_cos / e_addr_cos of every candidate pair.
Typo'd true names score ~0.75-0.85 against ~0.4-0.6 for look-alikes.

    python embed.py            # vectors for train + test, then retrieval for both
"""
import time

import faiss
import numpy as np
import polars as pl
from model2vec import StaticModel

import config

MODEL = "minishlab/potion-multilingual-128M"
DIM = 64
TOP = 5
MIN_COS = 0.5


def emb_path(split, k, what):
    return config.WORK_DIR / "emb" / f"{split}_s{k}_{what}.npy"


def em_path(split):
    return config.WORK_DIR / "cands" / f"{split}_em.parquet"


def _encode(model, texts, batch=200_000):
    out = np.zeros((len(texts), DIM), dtype=np.float16)
    for i in range(0, len(texts), batch):
        e = model.encode(texts[i:i + batch], batch_size=8192, use_multiprocessing=False).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        out[i:i + batch] = e
    return out


def build_vectors(splits=("train", "test"), force=False):
    t0 = time.time()
    model = StaticModel.from_pretrained(MODEL, dimensionality=DIM, force_download=False)
    for split in splits:
        for k in (1, 2, 3):
            f = pl.read_parquet(config.norm_path(split, k), columns=["core", "ad"])
            for what, col in (("name", "core"), ("addr", "ad")):
                path = emb_path(split, k, what)
                if path.exists() and not force:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, _encode(model, f[col].to_list()))
            print(f"  vectors {split} s{k}: {f.height} records ({time.time()-t0:.0f}s)", flush=True)


def load_vectors(split):
    """{"name"|"addr": {1|2|3: float16 array}} or None if not built."""
    if not emb_path(split, 1, "name").exists():
        return None
    return {w: {k: np.load(emb_path(split, k, w)) for k in (1, 2, 3)} for w in ("name", "addr")}


def retrieve(split, nprobe=24):
    """Top-5 Source-1 names for every Source-2/3 record without an address number."""
    t0 = time.time()
    s1 = pl.read_parquet(config.norm_path(split, 1), columns=["country"]).with_row_index("i1")
    v1 = np.load(emb_path(split, 1, "name")).astype(np.float32)
    rng = np.random.default_rng(config.SEED)
    out = []
    for country in sorted(s1["country"].unique().to_list()):
        ids1 = s1.filter(pl.col("country") == country)["i1"].to_numpy()
        xb = np.ascontiguousarray(v1[ids1])
        nlist = int(min(4096, max(64, 4 * np.sqrt(len(xb)))))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(DIM), DIM, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(xb[rng.choice(len(xb), min(len(xb), 50 * nlist), replace=False)])
        index.add(xb)
        index.nprobe = nprobe
        for k in (2, 3):
            f = pl.read_parquet(config.norm_path(split, k), columns=["country", "nums", "core"]).with_row_index("i2")
            q = f.filter((pl.col("country") == country) & (pl.col("nums") == "") & (pl.col("core") != ""))["i2"].to_numpy()
            if len(q) == 0:
                continue
            xq = np.ascontiguousarray(np.load(emb_path(split, k, "name"), mmap_mode="r")[q].astype(np.float32))
            D, I = index.search(xq, TOP)
            keep = (I >= 0) & (D >= MIN_COS)
            out.append(pl.DataFrame({"i1": ids1[I[keep]].astype(np.uint32),
                                     "src": np.full(keep.sum(), k, dtype=np.int8),
                                     "i2": np.repeat(q, TOP).reshape(-1, TOP)[keep].astype(np.uint32),
                                     "ecos": D[keep].astype(np.float32)}))
            print(f"  retrieval {split} {country} S{k}: {len(q)} queries ({time.time()-t0:.0f}s)", flush=True)
    em = pl.concat(out).unique(["i1", "src", "i2"])
    em.write_parquet(em_path(split))
    print(f"{split}: {em.height} embedding-retrieved pairs ({time.time()-t0:.0f}s)")
    return em


if __name__ == "__main__":
    import sys
    splits = sys.argv[1:] or ["train", "test"]
    build_vectors(splits)
    for sp in splits:
        retrieve(sp)
