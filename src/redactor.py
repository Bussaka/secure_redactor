import logging
import os
import re
from collections import Counter

import spacy

from src.name_rules import NOT_ENTITIES, NameRules

logger = logging.getLogger(__name__)

# en_core_web_lg + the name rules gave the fewest leaks and least over-redaction in benchmarks/name_redaction.py;
# the small model is the fallback when lg is not installed
DEFAULT_SPACY_MODEL = "en_core_web_lg"
FALLBACK_SPACY_MODEL = "en_core_web_sm"

class HybridRedactor:
    """Combines Rule-Based RegEx and Machine Learning NER for localized PII redaction."""

    # Deterministic patterns, applied in order. Longer/more specific formats run first so that
    # e.g. a phone number is never partially consumed by the shorter National ID pattern.
    REGEX_RULES = [
        # 1. Email Addresses
        ("[REDACTED_EMAIL]", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')),
        # 2. Kenyan Mobile Numbers (+254 / 254 / 0 prefix, 07xx or 01xx, optional space or dash separators)
        ("[REDACTED_PHONE_NUMBER]", re.compile(r'(?<![\w+])(?:\+?254|0)[\s-]?[17]\d{2}[\s-]?\d{3}[\s-]?\d{3}(?!\d)')),
        # 3. KRA PIN (e.g. A012345678Z)
        ("[REDACTED_KRA_PIN]", re.compile(r'\b[AP]\d{9}[A-Z]\b')),
        # 4. Kenyan National ID Patterns (7 or 8 digit standalone numbers; older IDs have 7 digits)
        ("[REDACTED_NATIONAL_ID]", re.compile(r'\b\d{7,8}\b')),
    ]

    NER_LABELS = ("PERSON", "ORG", "GPE")

    def __init__(self, model_name: str = None, use_name_rules: bool = True):
        # The SPACY_MODEL environment variable selects a larger model (e.g. en_core_web_lg) without code changes
        model_name = model_name or os.environ.get("SPACY_MODEL", "").strip() or DEFAULT_SPACY_MODEL
        self.model_name, self.nlp = model_name, None
        for candidate in dict.fromkeys([model_name, FALLBACK_SPACY_MODEL]):
            try:
                self.nlp = spacy.load(candidate)
                self.model_name = candidate
                break
            except OSError:
                logger.warning("spaCy language pack '%s' not found.", candidate)
        if self.nlp is None:
            logger.warning("No spaCy language pack available. Names, organisations and places will NOT be redacted.")
        # Kenyan-name list + context cues, to catch names the statistical model misses
        self.name_rules = NameRules() if use_name_rules else None

    def redact_text(self, text: str) -> str:
        """Executes the dual-engine pipeline to mask sensitive data entirely within RAM."""
        return self.redact_with_report(text)[0]

    def redact_with_report(self, text: str):
        """Same as redact_text, but also returns a Counter of how many items of each PII type were masked."""
        counts = Counter()
        if not text.strip():
            return text, counts

        # --- Engine Pillar 1: Deterministic RegEx (Strict Structural Formats) ---
        # Runs first so NER can never claim part of an email or number and leave the rest exposed.
        for badge, pattern in self.REGEX_RULES:
            text, n = pattern.subn(badge, text)
            if n:
                counts[badge.strip("[]")] += n

        # --- Engine Pillar 2: Machine Learning (Contextual Names/Orgs/Locations) ---
        spans = []
        if self.nlp:
            for ent in self.nlp(text).ents:
                # Skip entities that are just our own badges being re-recognised
                if ent.label_ in self.NER_LABELS and "REDACTED_" not in ent.text:
                    # The rules also veto obvious non-names the model mislabels (e.g. "KSH" tagged as a PERSON)
                    if self.name_rules and all(w.strip(".,:") in NOT_ENTITIES for w in ent.text.lower().split()):
                        continue
                    spans.append((ent.start_char, ent.end_char, ent.label_))

        # --- Engine Pillar 3: Kenyan name rules (optional) ---
        if self.name_rules:
            spans.extend((start, end, "PERSON") for start, end in self.name_rules.find_spans(text)
                         if "REDACTED_" not in text[start:end])
            # Grow every person match over adjacent capitalised words (catches surnames the model skipped)
            spans = [(*self.name_rules.extend_person_span(text, start, end), label) if label == "PERSON"
                     else (start, end, label) for start, end, label in spans]
            # A name found once is redacted everywhere in the document ("...sent to Moraa's account")
            person_texts = [text[start:end] for start, end, label in spans if label == "PERSON"]
            spans.extend((start, end, "PERSON") for start, end in self.name_rules.find_repeats(text, person_texts))

        # Replace back to front so earlier character offsets stay valid as the text changes length
        for start, end, label in reversed(_merge_spans(spans)):
            badge = f"[REDACTED_{label}]"
            text = text[:start] + badge + text[end:]
            counts[badge.strip("[]")] += 1

        return text, counts

def _merge_spans(spans):
    """Combines overlapping entity spans into one. PERSON wins, because a name partly hidden is still a leak."""
    merged = []
    for start, end, label in sorted(spans):
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_label = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), "PERSON" if "PERSON" in (label, prev_label) else prev_label)
        else:
            merged.append((start, end, label))
    return merged

if __name__ == "__main__":
    redactor = HybridRedactor()
    sample = "Hello Mercy, please send the invoice to payment@safaricom.co.ke. My ID is 34567890 and phone is 0712345678."
    print("[*] Testing Redactor Outcome:\n")
    print(redactor.redact_text(sample))
