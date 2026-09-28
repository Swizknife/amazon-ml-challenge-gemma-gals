"""Re-decode the saved test scores with another thresholds file, without re-scoring.

    python redecode.py thresholds_oof.json
    python redecode.py thresholds.json --set France=0.9,0.7,0.6   # t_single,t_match,miss_mass

Rewrites output/matching_results.tsv only; candidate_pairs.tsv stays as written by
predict.py (same candidates, same model).
"""
import json
import sys

import polars as pl

import config
from data_io import write_id_lists
from predict import decode_by_country


def main(thr_file, overrides):
    scored = pl.read_parquet(config.artifact("scored_test", "parquet"))
    thr = json.loads((config.WORK_DIR / thr_file).read_text())
    for country, vals in overrides.items():
        ts, tm, mm = (float(v) for v in vals.split(","))
        thr[country] = dict(t_single=ts, t_match=tm, miss_mass=mm)
    s1 = pl.read_parquet(config.norm_path("test", 1), columns=["id", "country"]).with_row_index("i1")
    ids = {k: pl.read_parquet(config.norm_path("test", k), columns=["id"])["id"].to_list() for k in (2, 3)}
    pred = decode_by_country(scored, s1, thr)
    s1_ids = s1["id"].to_list()
    matches = {s1_ids[i1]: [ids[src][i2] for src, i2 in lst] for i1, lst in pred.items()}
    write_id_lists(config.OUT_DIR / "matching_results.tsv", "matched_entity_ids", s1_ids, matches)
    n = sum(len(v) for v in matches.values())
    print(f"re-decoded with {thr_file} {overrides or ''}: {len(matches)} S1 matched, {n / len(s1_ids):.2f} links per S1")


if __name__ == "__main__":
    args = sys.argv[1:]
    sets = dict(a.split("=", 1) for a in args[args.index("--set") + 1:]) if "--set" in args else {}
    main(args[0], sets)
