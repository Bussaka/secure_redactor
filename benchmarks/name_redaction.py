"""Benchmark: how well does each redaction setup remove Kenyan names, and how much ordinary text does it destroy?

Run:  python -m benchmarks.name_redaction [--models en_core_web_sm en_core_web_lg ...]

Measures, per configuration:
  * name recall      - share of name words (first names and surnames) removed from the output
  * people protected - share of name mentions where EVERY word of the name was removed
  * over-redaction   - share of ordinary business words (KSH, PAYE, Employer, Grace period...) wrongly removed
  * speed            - sentences per second after the model has loaded
"""
import argparse
import random
import re
import time

from src.name_rules import NameRules
from src.redactor import HybridRedactor

# Both names on the Kenyan names list
LISTED_PEOPLE = [
    ("Wanjiku", "Mwangi"), ("Otieno", "Ochieng"), ("Akinyi", "Odhiambo"), ("Kiprono", "Kiptoo"), ("Njeri", "Kamau"),
    ("Mutua", "Musyoka"), ("Chebet", "Rotich"), ("Wafula", "Wekesa"), ("Moraa", "Nyakundi"), ("Kawira", "Mugambi"),
    ("Kazungu", "Katana"), ("Halima", "Abdi"), ("Naserian", "Sankale"), ("Wairimu", "Njoroge"), ("Atieno", "Omondi"),
    ("Kibet", "Langat"), ("Nekesa", "Barasa"), ("Mwende", "Kilonzo"), ("Kerubo", "Momanyi"), ("Fatuma", "Hassan"),
    ("Jepkosgei", "Kosgei"), ("Nyambura", "Kariuki"), ("Adhiambo", "Owino"), ("Kioko", "Mutiso"), ("Ekai", "Lokorio"),
]
# Neither name is on the list: shows whether the rules generalise beyond their word list
UNLISTED_PEOPLE = [
    ("Sitienei", "Lagat"), ("Chepkurui", "Kibichii"), ("Nanjala", "Wekhomba"), ("Gakii", "Muthee"),
    ("Kalekye", "Mwongela"), ("Nthenya", "Ndanu"), ("Kanini", "Gichuki"), ("Jemeli", "Kiplimo"), ("Obura", "Ogendo"),
    ("Sagini", "Okindo"), ("Wairegi", "Ngatia"), ("Achola", "Otwori"), ("Chesang", "Lokwawi"), ("Namayi", "Ekidor"),
    ("Kanyua", "Mbaabu"), ("Wangeci", "Wairagu"), ("Lemaiyan", "Ntimama"), ("Mwanarusi", "Tsofa"),
    ("Dzame", "Kenga"), ("Ayuma", "Okwara"),
]
# Common English first name + Kenyan surname
MIXED_PEOPLE = [
    ("Peter", "Kamau"), ("Grace", "Wanjiru"), ("Brian", "Otieno"), ("Mercy", "Chebet"), ("Faith", "Mutua"),
    ("John", "Wafula"), ("Mary", "Achieng"), ("Dennis", "Kiprotich"), ("Esther", "Moraa"), ("Kevin", "Mwiti"),
]

CONTEXTS = [
    "This Employment Agreement is made between the Employer and {first} {last} (the Employee).",
    "Employee: {first} {last}    Staff No: 4471    Department: Finance",
    "Bill to: {first} {last}, P.O. Box 1234, Nairobi.",
    "Dear {first}, please find attached your payslip for March.",
    "Prepared by {first} {last}, Accounts Department.",
    "SURNAME: {LAST}   GIVEN NAMES: {FIRST}   SEX: F",
    "Received from {first} {last} the sum of KSH 25,000 being rent for April.",
    "{first} {last} has been appointed as the new branch accountant.",
    "Please contact Mr. {last} on extension 204 for any queries.",
    "The refund was approved by {first} {last} on Monday and sent to {first}'s account.",
    "Our records show that {first} joined the company in 2019.",
    "{FIRST} {LAST}\nP.O. BOX 4521-00100 NAIROBI",
    "Witnessed by: {first} {last}",
    "cc: {first} {last}; Finance Manager",
    "Thank you, {first}. Regards, Accounts Team",
    "The tenant, {first} {last}, agrees to pay rent on the 5th of every month.",
]

# Ordinary business sentences: every listed word must survive redaction
KEEP_SENTENCES = [
    ("Total due KSH 50,000 inclusive of VAT at 16%.", ["Total", "KSH", "VAT"]),
    ("Statutory deductions: PAYE, NSSF, NHIF and Housing Levy.", ["PAYE", "NSSF", "NHIF", "Housing", "Levy"]),
    ("Pay via M-Pesa Paybill 247247, account number INV-2041.", ["Pay", "M-Pesa", "Paybill"]),
    ("The Employer shall give the Employee one month written notice.", ["Employer", "Employee"]),
    ("Invoice Number 2041: Basic Salary and Net Pay are shown below.", ["Invoice", "Basic", "Salary", "Net", "Pay"]),
    ("Dear Sir/Madam, we acknowledge receipt of your application.", ["Dear", "Sir", "Madam"]),
    ("A Grace period of 14 days applies to late payments.", ["Grace"]),
    ("The Board approved the Annual Budget on Monday.", ["Board", "Annual", "Budget", "Monday"]),
    ("Kindly Confirm Receipt Of This Letter.", ["Kindly", "Confirm", "Receipt", "Letter"]),
    ("Please Submit The Signed Contract To Human Resources.", ["Submit", "Signed", "Contract", "Human", "Resources"]),
    ("PAYMENT TERMS: NET 30 DAYS", ["PAYMENT", "TERMS", "NET", "DAYS"]),
    ("Employee Benefits: Medical Cover, Pension and Annual Leave.", ["Benefits", "Medical", "Cover", "Pension", "Leave"]),
    ("Prepared by the Finance Department for the Audit Committee.", ["Finance", "Department", "Audit", "Committee"]),
    ("Payment Received With Thanks.", ["Payment", "Received", "Thanks"]),
    ("Name of Bank: Equity. Branch: Moi Avenue.", ["Name", "Bank", "Branch"]),
]

def build_cases(seed=11):
    """Each person appears in 3 randomly chosen contexts. Returns [(group, sentence, [name words])]."""
    rng = random.Random(seed)
    cases = []
    for group, people in (("listed", LISTED_PEOPLE), ("unlisted", UNLISTED_PEOPLE), ("mixed", MIXED_PEOPLE)):
        for first, last in people:
            for template in rng.sample(CONTEXTS, 3):
                sentence = template.format(first=first, last=last, FIRST=first.upper(), LAST=last.upper())
                words = []
                if "{first}" in template: words.append(first)
                if "{last}" in template: words.append(last)
                if "{FIRST}" in template: words.append(first.upper())
                if "{LAST}" in template: words.append(last.upper())
                cases.append((group, sentence, words))
    return cases

def survives(word, text):
    return re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", text) is not None

def evaluate(redactor, cases):
    stats = {g: {"words": 0, "removed": 0, "people": 0, "protected": 0} for g in ("listed", "unlisted", "mixed")}
    leaks = []
    start = time.perf_counter()
    for group, sentence, words in cases:
        out = redactor.redact_text(sentence)
        removed = [w for w in words if not survives(w, out)]
        s = stats[group]
        s["words"] += len(words)
        s["removed"] += len(removed)
        s["people"] += 1
        s["protected"] += len(removed) == len(words)
        if len(removed) != len(words):
            leaks.append(out)

    kept_total = kept_ok = 0
    over_redacted = []
    for sentence, keep in KEEP_SENTENCES:
        out = redactor.redact_text(sentence)
        for word in keep:
            kept_total += 1
            if survives(word, out):
                kept_ok += 1
            else:
                over_redacted.append(word)
    elapsed = time.perf_counter() - start
    n_sentences = len(cases) + len(KEEP_SENTENCES)
    return stats, 1 - kept_ok / kept_total, over_redacted, n_sentences / elapsed, leaks

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+",
                        default=["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"])
    parser.add_argument("--show-leaks", action="store_true", help="Print sentences where a name survived")
    args = parser.parse_args(argv)

    listed = NameRules().names
    for first, last in UNLISTED_PEOPLE:
        assert first.lower() not in listed and last.lower() not in listed, f"{first} {last} is on the names list"

    cases = build_cases()
    print(f"{len(cases)} name sentences, {len(KEEP_SENTENCES)} business sentences\n")
    header = (f"| {'Configuration':<26} | {'Name recall':>11} | {'Listed':>7} | {'Unlisted':>8} | {'Mixed':>6} "
              f"| {'People fully protected':>22} | {'Over-redaction':>14} | {'Sentences/s':>11} |")
    print(header)
    print("|" + "|".join("-" * (len(c)) for c in header.split("|")[1:-1]) + "|")

    for model in args.models:
        for rules in (False, True):
            redactor = HybridRedactor(model_name=model, use_name_rules=rules)
            if redactor.nlp is None:
                print(f"| {model:<26} | not installed")
                break
            stats, over, over_words, speed, leaks = evaluate(redactor, cases)
            totals = {k: sum(s[k] for s in stats.values()) for k in ("words", "removed", "people", "protected")}
            pct = lambda s: f"{100 * s['removed'] / s['words']:.1f}%"
            name = model.replace("en_core_web_", "") + (" + name rules" if rules else "")
            print(f"| {name:<26} | {pct(totals):>11} | {pct(stats['listed']):>7} | {pct(stats['unlisted']):>8} "
                  f"| {pct(stats['mixed']):>6} | {100 * totals['protected'] / totals['people']:>21.1f}% "
                  f"| {100 * over:>13.1f}% | {speed:>11.0f} |")
            if args.show_leaks:
                for line in leaks:
                    print("    LEAK:", line.replace("\n", " / "))
                if over_words:
                    print("    OVER-REDACTED:", ", ".join(over_words))

if __name__ == "__main__":
    main()
