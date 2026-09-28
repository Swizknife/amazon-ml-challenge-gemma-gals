"""Indian-script names/addresses -> Latin text.

Names written in Devanagari, Tamil, Telugu, ... in this data are word-by-word
renderings of the English name (e.g. "श्याम कंसल्टिंग प्रा. लि." = "Shyam Consulting
Pvt Ltd"). We mine a token dictionary from *training* matches by aligning such a
name with its Source-1 owner's name position by position; words the dictionary
has not seen fall back to rule-based transliteration (ITRANS).
"""
import collections
import json
import re

import polars as pl
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

INDIC = re.compile("[ऀ-ൿ]")
INDIC_CLASS = "[ऀ-ൿ]"
ZW = re.compile("[​-‍﻿]")
TOKEN = re.compile(r"[^\s,.()\-]+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Unicode blocks of 0x80 code points starting at U+0900, in order.
_SCRIPTS = [sanscript.DEVANAGARI, sanscript.BENGALI, sanscript.GURMUKHI, sanscript.GUJARATI,
            sanscript.ORIYA, sanscript.TAMIL, sanscript.TELUGU, sanscript.KANNADA,
            sanscript.MALAYALAM]
_fallback_cache = {}


def fallback(tok):
    """Rule-based transliteration of one Indian-script token to lowercase ASCII."""
    out = _fallback_cache.get(tok)
    if out is None:
        idx = next(((ord(c) - 0x0900) // 0x80 for c in tok if 0x0900 <= ord(c) <= 0x0D7F), None)
        try:
            out = transliterate(tok, _SCRIPTS[idx], sanscript.ITRANS) if idx is not None else tok
        except Exception:
            out = ""
        out = _NON_ALNUM.sub("", out.lower())
        _fallback_cache[tok] = out
    return out


def _token(tok, tdict):
    return (tdict.get(tok) or fallback(tok)) if INDIC.search(tok) else tok


def translate_name(s, tdict):
    return TOKEN.sub(lambda m: _token(m.group(), tdict), s)


def translate_addr(s, adict, tdict):
    """Indian-script address segments are almost always state names: map whole
    segments via the mined segment dictionary, else translate token by token."""
    segs = []
    for seg in s.split(","):
        seg = seg.strip()
        if INDIC.search(seg):
            seg = adict.get(seg) or translate_name(seg, tdict)
        segs.append(seg)
    return ", ".join(segs)


def mine(s1, others, truth, min_share=0.5):
    """Build (token dict, address-segment dict) from training matches only."""
    owner = {x: s for s, ids in truth.items() for x in ids}
    s1_name = dict(zip(s1["entity_id"].to_list(), s1["business_name"].to_list()))
    s1_addr = dict(zip(s1["entity_id"].to_list(), s1["business_address"].to_list()))
    tok_co = collections.defaultdict(collections.Counter)
    seg_co = collections.defaultdict(collections.Counter)
    for df in others:
        sub = df.filter(pl.col("business_name").str.contains(INDIC_CLASS)
                        | pl.col("business_address").str.contains(INDIC_CLASS))
        for eid, name, addr in sub.select("entity_id", "business_name", "business_address").iter_rows():
            s = owner.get(eid)
            if s is None:
                continue
            if INDIC.search(name):
                nt = TOKEN.findall(ZW.sub("", name))
                et = [t.lower() for t in TOKEN.findall(s1_name[s])]
                if len(nt) == len(et):
                    for a, b in zip(nt, et):
                        if INDIC.search(a):
                            tok_co[a][b] += 1
            if INDIC.search(addr):
                s1_segs = {x.strip().lower() for x in s1_addr[s].split(",")}
                for g in ZW.sub("", addr).split(","):
                    g = g.strip()
                    if INDIC.search(g):
                        seg_co[g].update(s1_segs)

    def best(co, min_count):
        out = {}
        for k, c in co.items():
            tot = sum(c.values())
            v, n = c.most_common(1)[0]
            if tot >= min_count and n / tot >= min_share:
                out[k] = v
        return out

    # Segment co-occurrence counts every S1 segment, so use the top one without a share cut.
    adict = {g: c.most_common(1)[0][0] for g, c in seg_co.items() if sum(c.values()) >= 3}
    return best(tok_co, 2), adict


def save(path, tdict, adict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tokens": tdict, "segments": adict}, ensure_ascii=False), encoding="utf-8")


def load(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["tokens"], d["segments"]
