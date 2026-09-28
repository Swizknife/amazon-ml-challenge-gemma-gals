"""Error analysis on the holdout: dump wrong merges and missed matches with the
raw records so a human can spot patterns (python analysis.py).

Writes work/errors_fp.tsv (predicted, not true) and work/errors_fn.tsv (true,
not predicted; `reason` says whether blocking missed it or the model scored it low).
"""
import json

import polars as pl

import config
from data_io import read_ground_truth
from decode import decode
from metric import f05


def main(n=400):
    hold = pl.read_parquet(config.artifact("holdout_pred", "parquet"))
    ho_ids = pl.read_parquet(config.artifact("holdout_ids", "parquet"))["i1"].to_list()
    thr = json.loads(config.artifact("thresholds", "json").read_text())
    s1 = pl.read_parquet(config.norm_path("train", 1)).select("id", "business_name", "business_address", "country").with_row_index("i1")
    so = pl.concat([pl.read_parquet(config.norm_path("train", k)).select("id", "business_name", "business_address")
                    .with_row_index("i2").with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    country = dict(zip(s1["i1"].to_list(), s1["country"].to_list()))
    pred = {}
    for c in set(country[i] for i in ho_ids):
        t = thr.get(c, thr["_default"])
        ids = [i for i in ho_ids if country[i] == c]
        pred.update(decode(hold.filter(pl.col("i1").is_in(ids)), t["t_single"], t["t_match"], t["miss_mass"]))
    s1_id = s1["id"].to_list()
    truth = read_ground_truth(config.DATA_DIR / "train" / "train_ground_truth.tsv")
    id_of = {(k, i): x for k in (2, 3) for i, x in enumerate(so.filter(pl.col("src") == k)["id"].to_list())}
    key_of = {v: k for k, v in id_of.items()}
    scored = {(a, b, c): p for a, b, c, p in hold.select("i1", "src", "i2", "p").iter_rows()}

    fp, fn, per_c = [], [], {}
    for i in ho_ids:
        true = {key_of[x] for x in truth[s1_id[i]]}
        got = set(pred.get(i, []))
        per_c.setdefault(country[i], []).append(f05(got, true))
        fp += [(i, s, j, scored.get((i, s, j)), "wrong merge" if true else "singleton trap") for s, j in got - true]
        fn += [(i, s, j, scored.get((i, s, j)), "blocking miss" if (i, s, j) not in scored else "model low")
               for s, j in true - got]
    for c, v in per_c.items():
        print(f"{c}: holdout F0.5 {sum(v)/len(v):.4f} over {len(v)} entities")
    print(f"false positives {len(fp)}, false negatives {len(fn)} "
          f"(blocking misses {sum(r[4] == 'blocking miss' for r in fn)})")

    rec = so.select("src", "i2", "business_name", "business_address")
    base = s1.select("i1", s1_name="business_name", s1_address="business_address", country="country")
    for name, rows in (("errors_fp.tsv", fp), ("errors_fn.tsv", fn)):
        df = pl.DataFrame(rows, schema=["i1", "src", "i2", "p", "reason"], orient="row").sample(
            n=min(n, len(rows)), seed=config.SEED)
        df = (df.with_columns(pl.col("i1").cast(pl.UInt32), pl.col("src").cast(pl.Int8), pl.col("i2").cast(pl.UInt32))
              .join(base, on="i1").join(rec, on=["src", "i2"]).sort("reason"))
        df.write_csv(config.WORK_DIR / name, separator="\t")
        print(f"wrote work/{name} ({df.height} rows)")


def blocking_misses(n=25, country=None):
    """Print normalized forms of true pairs that blocking did not produce."""
    from blocking import _truth_pairs
    truth = _truth_pairs("train", None)
    cands = pl.read_parquet(config.WORK_DIR / "cands" / "train.parquet").select("i1", "src", "i2")
    miss = truth.join(cands, on=["i1", "src", "i2"], how="anti")
    cols = ["business_name", "core", "business_address", "ad", "nums", "country"]
    s1 = pl.read_parquet(config.norm_path("train", 1)).select(cols).with_row_index("i1")
    so = pl.concat([pl.read_parquet(config.norm_path("train", k)).select(cols[:-1]).with_row_index("i2")
                    .with_columns(src=pl.lit(k, pl.Int8)) for k in (2, 3)])
    miss = miss.join(s1, on="i1").join(so, on=["src", "i2"], suffix="_2")
    if country:
        miss = miss.filter(pl.col("country") == country)
    print(f"missed true pairs: {miss.height}")
    for r in miss.sample(n=min(n, miss.height), seed=config.SEED).iter_rows(named=True):
        print(f"S1 [{r['core']}] [{r['ad']}] [{r['nums']}]  <-  {r['business_name']} | {r['business_address']}")
        print(f"S{r['src']} [{r['core_2']}] [{r['ad_2']}] [{r['nums_2']}]  <-  {r['business_name_2']} | {r['business_address_2']}\n")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["misses"]:
        blocking_misses(country=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main()
