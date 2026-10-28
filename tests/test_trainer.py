import json
import os
import pickle
import random

import joblib
import pytest
from cryptography.fernet import Fernet

import main
import src.classifier as classifier_module
from src.classifier import DocumentClassifier, UNKNOWN_LABEL, metadata_path
from src.trainer import TrainingDataError, load_training_documents, train_classifier

FIRST_NAMES = ["Wanjiku", "Otieno", "Achieng", "Kiprono", "Njeri", "Mutua", "Akinyi", "Kamau", "Chebet", "Odhiambo",
               "Wairimu", "Kipchoge", "Nyambura", "Omondi", "Jepkosgei", "Mwende", "Barasa", "Wafula", "Moraa", "Nduta",
               "Kibet", "Auma", "Githinji", "Muthoni", "Ouma", "Cherono", "Njoroge", "Adhiambo", "Korir", "Waweru",
               "Nekesa", "Kimani", "Anyango", "Rotich", "Mumbua", "Onyango"]
LAST_NAMES = ["Mwangi", "Ochieng", "Atieno", "Kiptoo", "Kariuki", "Musyoka", "Wekesa", "Oduya", "Kirui", "Maina",
              "Owino", "Langat", "Njuguna", "Okoth", "Kosgei", "Mutiso", "Simiyu", "Juma", "Nyaga", "Ruto",
              "Kinyua", "Opiyo", "Cheruiyot", "Gitau", "Obiero", "Ngetich", "Wambua", "Otieno", "Kemboi", "Macharia",
              "Wanyama", "Kahiga", "Okello", "Bett", "Kilonzo", "Omollo"]
# One distinct person per document, like real payslips and contracts
NAMES = [f"{first} {last}" for first, last in zip(FIRST_NAMES, LAST_NAMES)]

TEMPLATES = {
    "Invoice": [
        "TAX INVOICE No {n}. Bill to {name}. Amount due KSH {amount}. Payment terms 30 days. "
        "Bank transfer to account {acct}. VAT 16% included. Thank you for your business.",
        "Invoice for supply of goods to {name}. Subtotal KSH {amount}, VAT, total due. "
        "Remit payment by bank transfer or M-Pesa paybill. Invoice number {n}.",
    ],
    "Contract": [
        "EMPLOYMENT CONTRACT between the Employer and {name} (the Employee), ID {id}. "
        "Position: accountant. Basic salary KSH {amount} per month. Probation three months. Termination notice one month.",
        "This Employment Agreement is made with {name}, phone {phone}. The employee is entitled to annual leave, "
        "medical cover and pension. Either party may terminate this agreement with written notice.",
    ],
    "Payslip": [
        "PAYSLIP for the month of March. Employee: {name}. Staff ID {id}. Basic pay KSH {amount}. "
        "Deductions: PAYE, NSSF, NHIF, housing levy. Net pay credited to account {acct}.",
        "Monthly pay statement for {name}. Gross earnings KSH {amount}. Statutory deductions PAYE NHIF NSSF. "
        "Net salary. Pay period ending 31st. Employee number {id}.",
    ],
}

def fake_document(label, rng, name=None):
    return rng.choice(TEMPLATES[label]).format(
        n=rng.randint(1000, 9999), name=name or rng.choice(NAMES), amount=f"{rng.randint(10, 900)},000",
        acct=rng.randint(10**11, 10**12), id=rng.randint(10**7, 10**8 - 1), phone=f"07{rng.randint(10**7, 10**8 - 1)}")

@pytest.fixture
def training_dir(tmp_path):
    rng = random.Random(7)
    root = tmp_path / "training"
    people = iter(NAMES)
    for label in TEMPLATES:
        (root / label).mkdir(parents=True)
        for i in range(12):
            doc = fake_document(label, rng, name=next(people))
            (root / label / f"{label.lower()}_{i:02d}.txt").write_text(doc, encoding="utf-8")
    return root

@pytest.fixture
def master_key():
    return Fernet.generate_key()

@pytest.fixture
def trained(training_dir, tmp_path, master_key, extractor, redactor):
    model_path = str(tmp_path / "models" / "document_classifier.joblib")
    metadata = train_classifier(str(training_dir), model_path, master_key, extractor, redactor)
    return model_path, metadata

def test_training_produces_a_working_signed_model(trained, master_key):
    model_path, metadata = trained
    assert metadata["labels"] == ["Contract", "Invoice", "Payslip"]
    assert metadata["samples_per_label"] == {"Contract": 12, "Invoice": 12, "Payslip": 12}
    assert metadata["evaluation"]["accuracy"] >= 0.9
    assert os.path.exists(model_path) and os.path.exists(metadata_path(model_path))

    clf = DocumentClassifier(model_path=model_path, master_key=master_key)
    assert clf.is_trained_model
    assert clf.predict_type("Payslip for April. Basic pay, deductions PAYE NSSF NHIF, net pay.") == "Payslip"
    assert clf.predict_type("Invoice number 55. Amount due, VAT, bank transfer payment.") == "Invoice"
    assert clf.predict_type("Lunch menu for Friday: chapati and beans") == UNKNOWN_LABEL

def test_saved_model_contains_no_pii(trained):
    model_path, _ = trained
    vocabulary = joblib.load(model_path).named_steps["tfidf"].vocabulary_
    words = " ".join(vocabulary)
    for name in NAMES:
        for part in name.lower().split():
            assert part not in words, f"name fragment '{part}' leaked into the model vocabulary"
    assert not any(ch.isdigit() for ch in words)

def test_tampered_model_falls_back_to_demo(trained, master_key):
    model_path, _ = trained
    with open(model_path, "ab") as f:
        f.write(b"tamper")
    clf = DocumentClassifier(model_path=model_path, master_key=master_key)
    assert not clf.is_trained_model

def test_edited_metadata_is_rejected(trained, master_key):
    model_path, _ = trained
    meta = json.loads(open(metadata_path(model_path), encoding="utf-8").read())
    meta["min_confidence"] = 0.0
    open(metadata_path(model_path), "w", encoding="utf-8").write(json.dumps(meta))
    assert not DocumentClassifier(model_path=model_path, master_key=master_key).is_trained_model

def test_wrong_master_key_is_rejected(trained):
    model_path, _ = trained
    assert not DocumentClassifier(model_path=model_path, master_key=Fernet.generate_key()).is_trained_model

class _Exploit:
    """A pickle payload that creates a marker file the moment it is unpickled."""
    def __init__(self, marker):
        self.marker = marker
    def __reduce__(self):
        return (open, (self.marker, "w"))

def test_malicious_pickle_is_never_executed(trained, master_key, tmp_path):
    model_path, _ = trained
    marker = tmp_path / "pwned.txt"
    with open(model_path, "wb") as f:
        pickle.dump(_Exploit(str(marker)), f)
    DocumentClassifier(model_path=model_path, master_key=master_key)
    assert not marker.exists()

def test_sklearn_version_mismatch_falls_back(trained, master_key, monkeypatch):
    model_path, _ = trained
    monkeypatch.setattr(classifier_module.sklearn, "__version__", "0.0.1")
    assert not DocumentClassifier(model_path=model_path, master_key=master_key).is_trained_model

def test_missing_model_uses_demo(tmp_path, master_key):
    clf = DocumentClassifier(model_path=str(tmp_path / "nope.joblib"), master_key=master_key)
    assert not clf.is_trained_model
    assert clf.predict_type("Tax Invoice details: Total due is KSH 12,000 payable to account standard bank.") == "Invoice"

def test_unreadable_files_are_skipped(training_dir, extractor, redactor):
    (training_dir / "Invoice" / "broken.docx").write_bytes(b"PK")
    texts, labels, skipped = load_training_documents(str(training_dir), extractor, redactor)
    assert skipped == [os.path.join("Invoice", "broken.docx")]
    assert len(texts) == len(labels) == 36

def test_training_text_is_redacted(training_dir, extractor, redactor):
    texts, _, _ = load_training_documents(str(training_dir), extractor, redactor)
    joined = " ".join(texts)
    assert "[REDACTED_NATIONAL_ID]" in joined and "[REDACTED_PERSON]" in joined
    # No National-ID-length digit runs survive (bank account numbers are not yet covered by a redaction rule)
    assert not any(len(run) in (7, 8) for run in "".join(c if c.isdigit() else " " for c in joined).split())

def test_all_kenyan_names_redacted(training_dir, extractor, redactor):
    # Failed with spaCy alone (en_core_web_sm missed e.g. "between the Employer and Wanjiku Mwangi");
    # the Kenyan name rules close the gap
    texts, _, _ = load_training_documents(str(training_dir), extractor, redactor)
    joined = " ".join(texts)
    leaked = [name for name in NAMES if name.split()[0] in joined]
    assert leaked == []

@pytest.mark.parametrize("layout, message", [
    ({"Invoice": 5}, "at least 2 document categories"),
    ({"Invoice": 5, "Contract": 1}, "at least 2 readable documents"),
])
def test_insufficient_training_data(tmp_path, extractor, redactor, master_key, layout, message):
    rng = random.Random(1)
    for label, n in layout.items():
        (tmp_path / "t" / label).mkdir(parents=True)
        for i in range(n):
            (tmp_path / "t" / label / f"{i}.txt").write_text(fake_document(label, rng), encoding="utf-8")
    with pytest.raises(TrainingDataError, match=message):
        train_classifier(str(tmp_path / "t"), str(tmp_path / "m.joblib"), master_key, extractor, redactor)

def test_train_command_end_to_end(pipeline_dirs, training_dir, monkeypatch):
    import shutil
    shutil.copytree(training_dir, pipeline_dirs["TRAINING_DIR"])
    assert main.train_model(str(pipeline_dirs["TRAINING_DIR"]), 0.5) == 0

    audit = [json.loads(l) for l in (pipeline_dirs["AUDIT_DIR"] / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert audit[-1]["event"] == "model_trained"
    assert audit[-1]["labels"] == ["Contract", "Invoice", "Payslip"]

    # The watcher now picks up the trained model and can recognise a category the demo model never knew
    from tests.test_pipeline import run_loop
    (pipeline_dirs["INCOMING_DIR"] / "slip.txt").write_text(
        "PAYSLIP for May. Employee: Jane Doe. Basic pay KSH 80,000. Deductions PAYE NSSF NHIF. Net pay.", encoding="utf-8")
    run_loop(monkeypatch, scans=2)
    [safe] = os.listdir(pipeline_dirs["SAFE_DIR"])
    assert (pipeline_dirs["SAFE_DIR"] / safe).read_text(encoding="utf-8").startswith("--- DOCUMENT CATEGORY: PAYSLIP ---")

def test_train_command_reports_missing_data(pipeline_dirs):
    assert main.train_model(str(pipeline_dirs["TRAINING_DIR"]), 0.5) == 1
