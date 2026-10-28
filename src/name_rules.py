import os
import re

NAMES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "kenyan_names.txt")

# A capitalised word ("Wanjiku", "O'Brien", "Mary-Anne") or an ALL-CAPS word ("MWANGI", as printed on ID cards)
_NAME_WORD = r"(?:[A-Z][a-z]+(?:['\-][A-Za-z]+)*|[A-Z]{2,}(?:['\-][A-Z]+)*)"
_NAME_SEQUENCE = re.compile(rf"{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,3}}")

# Phrases that are almost always followed by a person's name in business documents.
# Form labels ("Employee:", "Signed:") only count with a colon or dash, so "the Signed Contract" is not a cue;
# natural-language phrases ("Dear", "prepared by", "Mr.") count as they are.
_CUE = re.compile(r"""(?ix)
    \b(?:
        (?:
            (?:full\ )?names? | surname | given\ names | other\ names | employee(?:\ name)? | staff\ name
          | customer(?:\ name)? | client(?:\ name)? | holder(?:'s)?\ name | payee | attention | attn | cc | bcc
          | signed | signature | witness | contact\ person | next\ of\ kin | tenant | landlord | guarantor
        )[ \t]*[:\-][ \t]*
      | (?:
            (?:mr|mrs|ms|miss|dr|prof|eng|hon|rev)\.? | dear | bill(?:ed)?\ to:? | received\ from | paid\ to
          | (?:signed|prepared|approved|authori[sz]ed|checked|witnessed|issued|verified)\ by:?
          | between\ the\ employer\ and | the\ tenant,
        )[ \t]+
    )
""")

# Capitalised words that are not names; they stop a name sequence (e.g. "Kamau Holdings Ltd" -> "Kamau")
STOP_WORDS = {
    "ltd", "limited", "plc", "llp", "inc", "co", "company", "group", "holdings", "enterprises", "investments",
    "traders", "agencies", "sacco", "bank", "school", "hospital", "county", "ministry", "university", "college",
    "id", "no", "number", "date", "dob", "sex", "male", "female", "district", "of", "birth", "place", "issue",
    "serial", "signature", "kenya", "republic", "nairobi", "the", "and", "employee", "employer", "invoice",
    "payslip", "contract", "agreement", "account", "tel", "phone", "email", "mobile", "ksh", "kes", "vat",
    "paye", "nssf", "nhif", "sir", "madam", "sirs", "all", "team", "manager", "director", "accounts", "hr",
    "position", "title", "department", "dept", "staff", "page", "total", "amount", "net", "gross", "basic",
    "name", "names", "surname", "given", "other", "full", "nationality", "address", "box", "cc", "bcc",
    # Honorifics and greetings stay readable; only the name after them is redacted
    "mr", "mrs", "ms", "miss", "dr", "prof", "eng", "hon", "rev", "dear", "attn", "attention",
}

# Kenyan business terms the statistical model sometimes mislabels as people or organisations
NOT_ENTITIES = {"ksh", "kes", "vat", "paye", "nssf", "nhif", "shif", "kra", "pin", "m-pesa", "mpesa", "paybill",
                "till", "housing levy", "id", "id no"}

def _load_names(path: str):
    with open(path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}

class NameRules:
    """Rule-based person-name finder that complements spaCy NER for Kenyan documents.

    Two signals: (1) a capitalised word from the Kenyan names list, extended to neighbouring capitalised words
    so unlisted surnames are caught too; (2) a capitalised sequence right after a cue such as "Mr." or "Employee:".
    """

    def __init__(self, names_path: str = NAMES_FILE):
        self.names = _load_names(names_path)

    def find_spans(self, text: str):
        """Returns sorted, non-overlapping (start, end) character spans that look like person names."""
        spans = []

        # Signal 1: a listed name inside a run of capitalised words
        for match in _NAME_SEQUENCE.finditer(text):
            words = match.group().split()
            if any(_base(w) in self.names for w in words):
                # Cut the run at the first stop word so "Otieno Holdings Ltd" keeps only "Otieno"
                trimmed = self._trim_inner(text, match.start(), match.end())
                if trimmed:
                    spans.append(trimmed)

        # Signal 2: whatever capitalised sequence follows a name cue
        for cue in _CUE.finditer(text):
            following = _NAME_SEQUENCE.match(text, cue.end())
            if following:
                trimmed = self._trim_inner(text, following.start(), following.end())
                if trimmed:
                    spans.append(trimmed)

        return _merge(spans)

    def find_repeats(self, text: str, person_texts):
        """Every other occurrence (any case, including possessives) of the words of already-detected names."""
        words = {w.lower() for name in person_texts for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", name)
                 if len(w) >= 3 and w.lower() not in STOP_WORDS}
        if not words:
            return []
        pattern = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True))
                             + r")(?:'s)?(?![\w-])", re.IGNORECASE)
        # Only capitalised occurrences: "Mutua" and "MUTUA" are names, a lowercase "grace" is a word
        return [(m.start(), m.end()) for m in pattern.finditer(text) if m.group()[0].isupper()]

    def extend_person_span(self, text: str, start: int, end: int):
        """Grows a detected name over neighbouring capitalised words on the same line, so a model that found
        only "Ayuma" in "Ayuma Okwara" does not leave the surname exposed."""
        before = re.search(rf"(?:{_NAME_WORD}[ \t]+)+$", text[:start])
        if before:
            words = list(re.finditer(rf"{_NAME_WORD}", before.group()))
            # Walk back word by word until a stop word, so "Dear Ayuma" stays "Dear [name]"
            for word in reversed(words):
                if word.group().lower() in STOP_WORDS:
                    break
                start = before.start() + word.start()
        after = re.match(rf"(?:[ \t]+{_NAME_WORD})+", text[end:])
        if after:
            for word in re.finditer(rf"{_NAME_WORD}", after.group()):
                if word.group().lower() in STOP_WORDS:
                    break
                end = end + word.end()
        return start, end

    def _trim_inner(self, text: str, start: int, end: int):
        """Keeps only the part of the run containing a name: stop words split the run into pieces."""
        pieces, piece_start, piece_end = [], None, None
        for word in re.finditer(r"\S+", text[start:end]):
            if word.group().lower() in STOP_WORDS:
                if piece_start is not None:
                    pieces.append((piece_start, piece_end))
                piece_start = None
                continue
            if piece_start is None:
                piece_start = start + word.start()
            piece_end = start + word.end()
        if piece_start is not None:
            pieces.append((piece_start, piece_end))
        if not pieces:
            return None
        # Prefer the piece containing a listed name; otherwise the first piece (the one right after a cue)
        for piece in pieces:
            if any(_base(w) in self.names for w in text[piece[0]:piece[1]].split()):
                return piece
        return pieces[0]

def _base(word: str) -> str:
    """Lower-cased word without a possessive, so "Moraa's" matches the listed name "Moraa"."""
    word = word.lower()
    return word[:-2] if word.endswith("'s") else word

def _merge(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
