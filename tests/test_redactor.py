import pytest

@pytest.mark.parametrize("text, badge", [
    # Emails, including ones NER would otherwise split
    ("Email jane.wanjiku@safaricom.co.ke today", "[REDACTED_EMAIL]"),
    ("Write to payment@acme.co.ke.", "[REDACTED_EMAIL]"),
    # Phone numbers in every common Kenyan layout
    ("Call 0712345678 now", "[REDACTED_PHONE_NUMBER]"),
    ("Call 0712 345 678 now", "[REDACTED_PHONE_NUMBER]"),
    ("Call 0712-345-678 now", "[REDACTED_PHONE_NUMBER]"),
    ("Call +254712345678 now", "[REDACTED_PHONE_NUMBER]"),
    ("Call +254 712 345 678 now", "[REDACTED_PHONE_NUMBER]"),
    ("Call 254712345678 now", "[REDACTED_PHONE_NUMBER]"),
    ("Call 0112345678 now", "[REDACTED_PHONE_NUMBER]"),
    # National IDs, old (7-digit) and new (8-digit)
    ("ID 34567890 attached", "[REDACTED_NATIONAL_ID]"),
    ("ID 1234567 attached", "[REDACTED_NATIONAL_ID]"),
    # KRA PINs
    ("KRA PIN A012345678Z on file", "[REDACTED_KRA_PIN]"),
    ("KRA PIN P051234567Q on file", "[REDACTED_KRA_PIN]"),
])
def test_structured_pii_is_fully_masked(redactor, text, badge):
    result = redactor.redact_text(text)
    assert badge in result
    # No digit run or address fragment from the original value may survive
    assert "@" not in result
    assert not any(ch.isdigit() for ch in result.replace("REDACTED", ""))

def test_person_names_are_masked(redactor):
    result = redactor.redact_text("The main point of contact is project director John Kamau.")
    assert "John" not in result and "Kamau" not in result
    assert "[REDACTED_PERSON]" in result

def test_combined_document(redactor):
    text = "Supplier onboarding: John Kamau, email john@kamau.co.ke, phone 0712 345 678, ID 34567890."
    result = redactor.redact_text(text)
    for leaked in ("John", "Kamau", "john@", "0712", "345 678", "34567890"):
        assert leaked not in result

def test_badges_are_not_re_redacted(redactor):
    result = redactor.redact_text("Email a@b.co.ke and call 0712345678.")
    assert "[[" not in result
    assert "REDACTED_ORG" not in result

@pytest.mark.parametrize("text", [
    "Amount KSH 12,500,000 paid.",
    "Invoice total 4500 due in 30 days.",
])
def test_non_pii_numbers_are_kept(redactor, text):
    assert redactor.redact_text(text) == text

def test_blank_text_is_returned_unchanged(redactor):
    assert redactor.redact_text("   ") == "   "

def test_redaction_report_counts_each_type(redactor):
    text, counts = redactor.redact_with_report("Mail a@b.co.ke or c@d.co.ke, call 0712345678, ID 34567890.")
    assert counts["REDACTED_EMAIL"] == 2
    assert counts["REDACTED_PHONE_NUMBER"] == 1
    assert counts["REDACTED_NATIONAL_ID"] == 1
    assert text.count("[REDACTED_EMAIL]") == 2
