"""Build <team>_submission.zip in the layout required by the challenge README.

    python make_submission.py --team TEAMNAME [--suffix 1]   # -> TEAMNAME_submission[1].zip

Layout:
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/{src/, README.md, requirements.txt}
  Documentation_template.md   (taken from submission/)
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code" / "business_entity_resolution"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--suffix", default="", help="keeps an earlier zip, e.g. 1 -> TEAM_submission1.zip")
    args = ap.parse_args()
    target = ROOT / f"{args.team}_submission{args.suffix}.zip"
    files = [(ROOT / "output" / n, f"output/{n}") for n in ("matching_results.tsv", "candidate_pairs.tsv")]
    files += [(p, f"code/business_entity_resolution/src/{p.name}") for p in sorted((CODE / "src").glob("*.py"))]
    files += [(CODE / n, f"code/business_entity_resolution/{n}") for n in ("README.md", "requirements.txt")]
    files.append((ROOT / "submission" / "Documentation_template.md", "Documentation_template.md"))
    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        raise SystemExit("missing files:\n  " + "\n  ".join(missing))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path, arc in files:
            z.write(path, arc)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
