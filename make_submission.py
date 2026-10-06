"""Build <team>_submission.zip in the layout required by the challenge README.

    python make_submission.py --team gemma_gals --out-dir <folder with both TSVs> \
        --doc <filled Documentation_template.md> [--uploaded <the file uploaded to the portal>] [--suffix 3]

Layout:
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/{src/ (all source, recursively), README.md, requirements.txt}
  Documentation_template.md

--uploaded checks, by hash, that output/matching_results.tsv is the very file uploaded to the
leaderboard (the rules require them to be identical); the zip is not built if they differ.
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code" / "business_entity_resolution"
SKIP_DIRS = {"__pycache__"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out-dir", required=True, help="folder holding matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--doc", required=True, help="the filled-in Documentation_template.md")
    ap.add_argument("--uploaded", default=None, help="the matching_results.tsv uploaded to the portal (hash check)")
    ap.add_argument("--suffix", default="", help="keeps earlier zips, e.g. 3 -> TEAM_submission3.zip")
    args = ap.parse_args()
    out_dir, doc = Path(args.out_dir), Path(args.doc)
    target = ROOT / f"{args.team}_submission{args.suffix}.zip"

    files = [(out_dir / n, f"output/{n}") for n in ("matching_results.tsv", "candidate_pairs.tsv")]
    src = CODE / "src"
    files += [(p, f"code/business_entity_resolution/{p.relative_to(CODE).as_posix()}")
              for p in sorted(src.rglob("*")) if p.is_file() and not (set(p.parts) & SKIP_DIRS) and p.suffix != ".pyc"]
    files += [(CODE / n, f"code/business_entity_resolution/{n}") for n in ("README.md", "requirements.txt")]
    files.append((doc, "Documentation_template.md"))

    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        raise SystemExit("missing files:\n  " + "\n  ".join(missing))
    if args.uploaded and md5(out_dir / "matching_results.tsv") != md5(args.uploaded):
        raise SystemExit("output/matching_results.tsv differs from the file uploaded to the leaderboard; "
                         "the rules require them to be identical")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path, arc in files:
            z.write(path, arc)
    n_src = sum(a.startswith("code/business_entity_resolution/src/") for _, a in files)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB, {len(files)} files, {n_src} under src/)")
    print(f"matching_results.tsv md5 {md5(out_dir / 'matching_results.tsv')}")


if __name__ == "__main__":
    main()
