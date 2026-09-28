"""Name/address normalization and the parallel driver that caches every source
as parquet (work/norm/{split}_s{k}.parquet).

Output columns per record:
  nm     all normalized name tokens          core   name without legal forms / honorifics
  legal  canonical legal forms (sorted)      cc     core with spaces removed (website names)
  script 1 if the raw name used an Indian script
  web    1 if the raw name looked like a website
  ad     address words (abbreviations expanded, unit words dropped)
  adf    full address tokens incl. number tokens, in order
  nums   every number in the address (leading zeros stripped), in order
  hn     first number token as its digit groups ("1577/15" -> "1577-15")
  hmain  longest digit group of the first number token
"""
import re
import time
import unicodedata
from multiprocessing import Pool

import polars as pl

import config
import rules
import translit
from data_io import read_ground_truth, read_source
from translit import INDIC, ZW

_LEGAL_PATTERNS = [(re.compile(p), r) for p, r in rules.LEGAL_PATTERNS]
_PIPE = re.compile(r"\|.*$")
_BRACKETS = re.compile(r"\[[^\]]*\]|\(\s*id\s*:?\s*\d+\s*\)")
_WEB = re.compile(r"\bwww\.|\.(?:%s)\b" % "|".join(rules.WEB_TLDS))
_MS = re.compile(r"\bm\s*/\s*s\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LEETABLE = re.compile(r"^(?:[013458][a-z]{2,}|[a-z]{2,}[013458]|[a-z]+[013458][a-z]+)$")
_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th)$")
_ADDR_TOK = re.compile(r"[a-z0-9]+(?:[/-][a-z0-9]+)*")
_DIGITS = re.compile(r"\d+")
_NUMERO = re.compile(r"\bn\s*[°º]")  # French "N° 52" -> "52"
_STATES = {**rules.US_STATES, **rules.IN_STATES}
_STATE_PHRASES = re.compile(r"\b(%s)\b" % "|".join(
    sorted((k for k in _STATES if " " in k), key=len, reverse=True)))
_STATE_WORDS = {k: v for k, v in _STATES.items() if " " not in k}

NAME_COLS = ["nm", "core", "legal", "cc", "script", "web"]
ADDR_COLS = ["ad", "adf", "nums", "hn", "hmain"]


def fold(s):
    """Lowercase and strip accents."""
    s = s.lower()
    if s.isascii():
        return s
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_name(raw, tdict):
    s = ZW.sub("", raw)
    script = 0
    if INDIC.search(s):
        script = 1
        s = translit.translate_name(s, tdict)
    s = _BRACKETS.sub(" ", _PIPE.sub(" ", fold(s)))
    web = 1 if _WEB.search(s) else 0
    s = _MS.sub(" ", _WEB.sub(" ", s))
    for pat, rep in _LEGAL_PATTERNS:
        s = pat.sub(rep, s)
    toks = []
    for t in _NON_ALNUM.sub(" ", s.replace("&", " and ")).split():
        if t.isdigit() and len(t) >= 4:  # record ids such as "#25015" are noise
            continue
        if _LEETABLE.match(t) and not _ORDINAL.match(t):
            t = t.translate(rules.LEET)
        if not toks or toks[-1] != t:  # "investments investments" -> once
            toks.append(t)
    legal = sorted({rules.LEGAL[t] for t in toks if t in rules.LEGAL})
    core = [t for t in toks if t not in rules.LEGAL and t not in rules.NAME_NOISE]
    if not core:  # name made only of legal words / noise
        core = [t for t in toks if t not in rules.NAME_NOISE] or toks
    return " ".join(toks), " ".join(core), " ".join(legal), "".join(core), script, web


def _num(d):
    return str(int(d)) if len(d) < 18 else d


def norm_addr(raw, adict, tdict):
    s = ZW.sub("", raw)
    if INDIC.search(s):
        s = translit.translate_addr(s, adict, tdict)
    s = _STATE_PHRASES.sub(lambda m: _STATES[m.group(1)], _NUMERO.sub(" ", fold(s)))
    words, full, nums = [], [], []
    hn = hmain = ""
    for t in _ADDR_TOK.findall(s):
        if any(c.isdigit() for c in t):
            ds = [_num(d) for d in _DIGITS.findall(t)]
            if not hn:
                hn, hmain = "-".join(ds), max(ds, key=len)
            nums.extend(ds)
            full.append(t)
        else:
            t = _STATE_WORDS.get(rules.ADDR_ABBR.get(t, t), rules.ADDR_ABBR.get(t, t))
            if t in rules.ADDR_DROP:
                continue
            words.append(t)
            full.append(t)
    return " ".join(words), " ".join(full), " ".join(nums), hn, hmain


_TD = _AD = None


def _init(tdict, adict):
    global _TD, _AD
    _TD, _AD = tdict, adict


def _work(chunk):
    names, addrs = chunk
    out = [[] for _ in NAME_COLS + ADDR_COLS]
    for n, a in zip(names, addrs):
        for col, v in zip(out, norm_name(n, _TD) + norm_addr(a, _AD, _TD)):
            col.append(v)
    return out


def normalize_frame(df, tdict, adict, workers=config.N_WORKERS, chunk=100_000):
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    with Pool(workers, initializer=_init, initargs=(tdict, adict)) as pool:
        parts = pool.map(_work, chunks)
    cols = NAME_COLS + ADDR_COLS
    data = {c: [v for p in parts for v in p[j]] for j, c in enumerate(cols)}
    schema = {c: (pl.Int8 if c in ("script", "web") else pl.String) for c in cols}
    return pl.concat([df.rename({"entity_id": "id"}), pl.DataFrame(data, schema=schema)],
                     how="horizontal")


def dict_path():
    return config.WORK_DIR / "translit.json"


def build(splits=("train", "test"), force=False):
    """Mine the Indian-script dictionaries (train only) and cache normalized sources."""
    t0 = time.time()
    if dict_path().exists() and not force:
        tdict, adict = translit.load(dict_path())
    else:
        raw = {k: read_source(config.src_path("train", k)) for k in (1, 2, 3)}
        truth = read_ground_truth(config.DATA_DIR / "train" / "train_ground_truth.tsv")
        tdict, adict = translit.mine(raw[1], [raw[2], raw[3]], truth)
        translit.save(dict_path(), tdict, adict)
        del raw
        print(f"dictionary: {len(tdict)} tokens, {len(adict)} address segments ({time.time()-t0:.0f}s)")
    for split in splits:
        for k in (1, 2, 3):
            out = config.norm_path(split, k)
            if out.exists() and not force:
                continue
            df = normalize_frame(read_source(config.src_path(split, k)), tdict, adict)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out)
            print(f"normalized {split} s{k}: {df.height} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    import sys
    build(force="--force" in sys.argv)
