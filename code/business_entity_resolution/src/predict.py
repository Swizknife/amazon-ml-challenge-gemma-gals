"""Score the test candidates and write matching_results.tsv + candidate_pairs.tsv."""
import json
import time

import lightgbm as lgb
import polars as pl

import blocking
import config
import features
import metablock
from data_io import write_id_lists
from decode import decode

if config.USE_EMB:
    import embed


def score_split(split, model):
    s1, so = features.load_records(split)
    cands = features.context(blocking.load_cands(split))  # context on the full blocking output
    if metablock.model_path().exists():
        cands = metablock.prune(cands, s1, so)  # supervised meta-blocking: the model scores only these
    emb = embed.load_vectors(split) if config.USE_EMB else None
    names = model.feature_name()
    parts = [X.select(*features.KEYS, p=pl.Series(model.predict(X.select(names).cast(pl.Float32).to_numpy()), dtype=pl.Float32))
             for X in features.build(cands, s1, so, emb=emb)]
    return s1, so, cands, pl.concat(parts)


def thresholds_file():
    """Most faithful tuning available: pruned full-universe OOF > full OOF > holdout."""
    for name in ("thresholds_final", "thresholds_oof", "thresholds"):
        path = config.artifact(name, "json")
        if path.exists():
            return path


def decode_by_country(scored, s1, thr):
    pred = {}
    for country in s1["country"].unique().to_list():
        t = thr.get(country, thr["_default"])
        i1s = s1.filter(pl.col("country") == country)["i1"]
        pred.update(decode(scored.filter(pl.col("i1").is_in(i1s)),
                           t["t_single"], t["t_match"], t["miss_mass"]))
    return pred


def main(split="test"):
    t0 = time.time()
    model = lgb.Booster(model_file=str(config.artifact("model", "txt")))
    thr = json.loads(thresholds_file().read_text())
    s1, so, cands, scored = score_split(split, model)
    scored.write_parquet(config.artifact(f"scored_{split}", "parquet"))
    print(f"scored {scored.height} pairs, {scored.height / s1.height:.2f} per S1, "
          f"thresholds from {thresholds_file().name} ({time.time()-t0:.0f}s)")

    ids = {k: so.filter(pl.col("src") == k).sort("i2")["id"].to_list() for k in (2, 3)}
    s1_ids = s1["id"].to_list()
    pred = decode_by_country(scored, s1, thr)
    matches = {s1_ids[i1]: [ids[src][i2] for src, i2 in lst] for i1, lst in pred.items()}
    cand_lists = (cands.join(so.select("src", "i2", "id"), on=["src", "i2"])
                  .group_by("i1").agg(pl.col("id")))
    candidates = {s1_ids[i1]: lst for i1, lst in cand_lists.iter_rows()}

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_id_lists(config.OUT_DIR / "matching_results.tsv", "matched_entity_ids", s1_ids, matches)
    write_id_lists(config.OUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, candidates)
    n_match = sum(len(v) for v in matches.values())
    print(f"wrote outputs: {len(matches)} of {len(s1_ids)} S1 matched, {n_match} links, "
          f"{n_match / len(s1_ids):.2f} per S1 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
