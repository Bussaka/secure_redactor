import pytest

from src.name_rules import NameRules

@pytest.fixture(scope="module")
def rules():
    return NameRules()

def found(rules, text):
    return [text[s:e] for s, e in rules.find_spans(text)]

@pytest.mark.parametrize("text, expected", [
    # Listed names, extended to an unlisted neighbour
    ("This agreement is between the Employer and Wanjiku Wairagu (the Employee).", ["Wanjiku Wairagu"]),
    ("Our records show that Akinyi joined in 2019.", ["Akinyi"]),
    # Unlisted names caught by context cues
    ("Employee: Sitienei Lagat    Staff No: 4471", ["Sitienei Lagat"]),
    ("Prepared by Achola Otwori, Accounts Department.", ["Achola Otwori"]),
    ("cc: Namayi Ekidor; Finance Manager", ["Namayi Ekidor"]),
    ("SURNAME: WEKHOMBA   GIVEN NAMES: NANJALA", ["WEKHOMBA", "NANJALA"]),
    # Honorifics and greetings stay readable
    ("Please contact Mr. Gichuki on extension 204.", ["Gichuki"]),
    ("Dear Kalekye, please find attached your payslip.", ["Kalekye"]),
    # Organisation words are cut off
    ("Payment from Otieno Holdings Ltd was received.", ["Otieno"]),
])
def test_names_are_found(rules, text, expected):
    assert found(rules, text) == expected

@pytest.mark.parametrize("text", [
    "Please Submit The Signed Contract To Human Resources.",
    "Dear Sir/Madam, we acknowledge receipt of your application.",
    "Statutory deductions: PAYE, NSSF, NHIF and Housing Levy.",
    "The Employer shall give the Employee one month written notice.",
    "PAYMENT TERMS: NET 30 DAYS",
    "Kindly Confirm Receipt Of This Letter.",
    "a lowercase mention of kamau is ignored",
])
def test_ordinary_text_is_left_alone(rules, text):
    assert found(rules, text) == []

def test_extend_person_span_covers_surname(rules):
    text = "Witnessed by: Ayuma Okwara today"
    start = text.index("Ayuma")
    s, e = rules.extend_person_span(text, start, start + len("Ayuma"))
    assert text[s:e] == "Ayuma Okwara"

def test_extend_stops_at_honorific_and_line_break(rules):
    text = "Dear Ayuma\nOkwara"
    start = text.index("Ayuma")
    s, e = rules.extend_person_span(text, start, start + len("Ayuma"))
    assert text[s:e] == "Ayuma"
