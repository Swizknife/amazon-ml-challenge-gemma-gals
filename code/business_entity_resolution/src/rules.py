"""Hand-written normalization rules (domain knowledge, no external data).

All maps are applied identically to every source, so a rule only has to make
both sides of a true pair look alike; it does not have to be linguistically
perfect.
"""

# --- names -----------------------------------------------------------------

# Legal-form tokens after punctuation removal -> canonical token.
LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc",
    "pc": "pc", "pa": "pa", "plc": "plc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci",
    "eurl": "eurl", "snc": "snc", "ei": "ei", "selarl": "selarl",
}

# Dotted / spaced legal forms that must be glued before tokenizing.
LEGAL_PATTERNS = [
    (r"\bs\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\b\.?", " sarl "),
    (r"\bs\s*\.\s*a\s*\.\s*s\s*\.?", " sas "),
    (r"\bs\s*\.\s*a\s*\.", " sa "),
    (r"\bl\s*\.\s*l\s*\.\s*c\s*\.?", " llc "),
    (r"\bl\s*\.\s*l\s*\.\s*p\s*\.?", " llp "),
    (r"\bp\s*\.\s*c\s*\.", " pc "),
    (r"\bpvt\s*-\s*ltd\b", " pvt ltd "),
]

# Tokens that are noise in names (honorifics, filler).
NAME_NOISE = {"mr", "mrs", "ms", "dr", "smt", "sri", "shri", "the", "of", "and", "et", "m/s"}

WEB_TLDS = ("com", "in", "net", "org", "co", "fr", "biz", "info", "us")

# Leetspeak-style digit substitutions seen inside words (e.g. "8eth").
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})

# --- addresses -------------------------------------------------------------

# Token -> canonical short form. Applied to single tokens after lowercasing.
ADDR_ABBR = {
    # street types (US / India / France)
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "avn": "ave",
    "road": "rd", "drive": "dr", "lane": "ln", "boulevard": "blvd", "bd": "blvd",
    "bvd": "blvd", "court": "ct", "circle": "cir", "place": "pl", "parkway": "pkwy",
    "highway": "hwy", "trail": "trl", "terrace": "ter", "terr": "ter", "square": "sq",
    "route": "rte", "rue": "rue", "r": "rue", "chemin": "chem", "ch": "chem",
    "impasse": "imp", "allee": "all", "quai": "qu", "cours": "crs",
    "point": "pt", "mount": "mt", "fort": "ft", "saint": "st", "sainte": "ste",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # Indian address words
    "near": "nr", "opposite": "opp", "industrial": "indl", "indl": "indl",
    "nagar": "ngr", "ngr": "ngr", "colony": "col", "sector": "sec", "sec": "sec",
    "building": "bldg", "bldg": "bldg", "apartment": "apt", "apartments": "apt",
    "apts": "apt", "floor": "flr", "fl": "flr", "flr": "flr",
    "bengaluru": "bangalore", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "gurugram": "gurgaon", "ahmadabad": "ahmedabad",
}

# Tokens dropped from addresses entirely (unit words, filler, postal noise).
ADDR_DROP = {
    "no", "nos", "number", "h", "hno", "house", "unit", "suite", "ste", "#", "po", "box",
    "c", "o", "cdp", "city", "of", "the", "de", "du", "des", "la", "le", "les",
    "d", "l", "cedex", "door", "plot", "flat", "shop", "dist", "district", "tal", "taluka",
    "vill", "village", "post", "at", "and", "&", "bis", "ter",
}

# Generic address words that are too common to be blocking keys.
ADDR_GENERIC = {
    "st", "ave", "rd", "dr", "ln", "blvd", "ct", "cir", "pl", "pkwy", "hwy", "trl", "ter",
    "sq", "rte", "rue", "chem", "imp", "all", "n", "s", "e", "w", "ne", "nw", "se", "sw",
    "nr", "opp", "main", "cross", "block", "sec", "ngr", "col", "bldg", "apt", "flr",
    "road", "ground", "first", "second", "phase", "stage", "layout", "area", "west", "east",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}

IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
}
