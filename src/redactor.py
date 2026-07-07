import re
import spacy

class HybridRedactor:
    """Combines Rule-Based RegEx and Machine Learning NER for localized PII redaction."""
    
    def __init__(self):
        # Load the lightweight local language model we downloaded earlier
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[-] Warning: spacy language pack 'en_core_web_sm' not found. Local names may be missed.")
            self.nlp = None

    def redact_text(self, text: str) -> str:
        """Executes the dual-engine pipeline to mask sensitive data entirely within RAM."""
        if not text.strip():
            return text

        # --- Engine Pillar 1: Machine Learning (Contextual Names/Orgs/Locations) ---
        if self.nlp:
            doc = self.nlp(text)
            # We sort entities in reverse order (back to front) to prevent shifting 
            # string character indices as we modify and replace text characters in memory.
            sorted_ents = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
            
            for ent in sorted_ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    # Mask standard entities with clean contextual badges
                    badge = f"[REDACTED_{ent.label_}]"
                    text = text[:ent.start_char] + badge + text[ent.end_char:]

        # --- Engine Pillar 2: Deterministic RegEx (Strict Structural Formats) ---
        
        # 1. Kenyan National ID Patterns (Strict 8-digit bounding strings)
        id_pattern = r'\b\d{8}\b'
        text = re.sub(id_pattern, "[REDACTED_NATIONAL_ID]", text)

        # 2. Kenyan Mobile Numbers (+254..., 07..., or 01... formats)
        phone_pattern = r'(?:\+254|0)[71]\d{8}\b'
        text = re.sub(phone_pattern, "[REDACTED_PHONE_NUMBER]", text)

        # 3. Email Addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, "[REDACTED_EMAIL]", text)

        return text

if __name__ == "__main__":
    redactor = HybridRedactor()
    sample = "Hello Mercy, please send the invoice to payment@safaricom.co.ke. My ID is 34567890 and phone is 0712345678."
    print("[*] Testing Redactor Outcome:\n")
    print(redactor.redact_text(sample))