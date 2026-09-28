"""Reading the challenge TSVs safely and writing submission files."""
import polars as pl

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def _count_rows(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def read_source(path):
    """Read a source TSV with every field as a plain string.

    quote_char=None keeps names containing '"' intact, and no null_values means
    literal strings such as "NA" stay strings instead of becoming missing.
    """
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                     missing_utf8_is_empty_string=True, encoding="utf8")
    assert df.columns == SOURCE_COLS, f"{path}: unexpected columns {df.columns}"
    expected = _count_rows(path)
    assert df.height == expected, f"{path}: parsed {df.height} rows, file has {expected}"
    return df.with_columns(pl.all().fill_null(""))


def read_ground_truth(path):
    """Return {source1_entity_id: [matched ids]} (empty list for singletons)."""
    truth = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            truth[s1] = [x for x in rest.split(",") if x]
    return truth


def write_id_lists(path, second_col, s1_ids, mapping):
    """Write one row per S1 id (in the given order) with a comma-joined id list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{second_col}\n")
        for s1 in s1_ids:
            ids = mapping.get(s1, ())
            f.write(f"{s1}\t{','.join(dict.fromkeys(ids))}\n")
