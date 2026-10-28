import pytest
from src.classifier import UNKNOWN_LABEL

@pytest.mark.parametrize("text, expected", [
    ("Tax Invoice details: Total due is KSH 12,000 payable to account standard bank.", "Invoice"),
    ("INVOICE No 0042. Bill to Acme Ltd. Amount due KSH 50,000. Payment via bank transfer.", "Invoice"),
    ("This Employment Agreement is between the Employer and the Employee. Basic salary KSH 80,000, 21 days leave.", "Contract"),
    ("Republic of Kenya National Identity Card. ID Number. Date of Birth. Sex: F. District of Birth: Nakuru.", "National ID"),
])
def test_known_document_types(classifier, text, expected):
    assert classifier.predict_type(text) == expected

@pytest.mark.parametrize("text", [
    "Lunch menu for Friday: chapati and beans",
    "Minutes of the board meeting held on Monday",
    "Please onboard our new supplier account for trade clearing. project director email mobile number ID card",
])
def test_unrelated_documents_are_unknown(classifier, text):
    assert classifier.predict_type(text) == UNKNOWN_LABEL

def test_empty_text_is_unknown(classifier):
    assert classifier.predict_type("  \n ") == UNKNOWN_LABEL
