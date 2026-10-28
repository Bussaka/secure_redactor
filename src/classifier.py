import hashlib
import hmac
import io
import json
import logging
import os
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "Unknown"

# A word must appear in at least this many training documents to enter the saved vocabulary. Names and other
# one-off PII that slip past redaction usually appear in only one or two documents, so they are never stored.
MIN_DOCUMENT_FREQUENCY = 3

def build_pipeline() -> Pipeline:
    """Text -> Mathematical Vectors -> Prediction, as used for models trained on real documents."""
    return Pipeline([
        # Letters-only tokens: digits (IDs, phone numbers, amounts) can never become part of the saved vocabulary.
        # Word pairs (bigrams) like "tax invoice" add useful context once there is real training data.
        ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
                                  ngram_range=(1, 2), sublinear_tf=True, min_df=MIN_DOCUMENT_FREQUENCY)),
        ('clf', MultinomialNB())
    ])

def build_demo_pipeline() -> Pipeline:
    """Simpler pipeline for the built-in synthetic demo model, which is too small for bigrams and holds no real data."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
        ('clf', MultinomialNB())
    ])

def _signing_key(master_key: bytes) -> bytes:
    # Separate derived key so model signatures never reuse the encryption key directly
    return hmac.new(master_key, b"secure-redactor-model-signing-v1", hashlib.sha256).digest()

def metadata_path(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".meta.json"

def _signature(master_key: bytes, model_bytes: bytes, metadata: dict) -> str:
    # Covers the metadata too, so settings like min_confidence cannot be edited without detection
    unsigned = {k: v for k, v in metadata.items() if k != "signature"}
    payload = model_bytes + b"\n" + json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_signing_key(master_key), payload, hashlib.sha256).hexdigest()

def save_model(model: Pipeline, model_path: str, master_key: bytes, metadata: dict) -> dict:
    """Serialises the trained pipeline and signs it, so a swapped or tampered model file is refused on load."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_bytes = buffer.getvalue()

    metadata = {
        **metadata,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
    }
    metadata["signature"] = _signature(master_key, model_bytes, metadata)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Write both files to temp names first so a crash never leaves a model without matching metadata
    with open(model_path + ".tmp", "wb") as f:
        f.write(model_bytes)
    with open(metadata_path(model_path) + ".tmp", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(model_path + ".tmp", model_path)
    os.replace(metadata_path(model_path) + ".tmp", metadata_path(model_path))
    return metadata

def load_model(model_path: str, master_key: bytes):
    """Returns (pipeline, metadata). Raises ValueError if the signature or library version does not match.

    The signature is checked BEFORE unpickling: loading a pickle can execute arbitrary code,
    so an unsigned or modified model file must never reach joblib.load.
    """
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    with open(metadata_path(model_path), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not hmac.compare_digest(_signature(master_key, model_bytes, metadata), metadata.get("signature", "")):
        raise ValueError("model signature mismatch (file was modified, or it was trained with a different master key)")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError(f"model was trained with scikit-learn {metadata.get('sklearn_version')}, "
                         f"but {sklearn.__version__} is installed; retrain with `python main.py train`")
    return joblib.load(io.BytesIO(model_bytes)), metadata

class DocumentClassifier:
    """Local, lightweight Machine Learning model to categorize corporate documents."""

    # Below these levels the model is effectively guessing, so we report "Unknown" instead of forcing a label.
    # A trained model stores its own confidence threshold in its metadata; this is the default for the seed model.
    MIN_CONFIDENCE = 0.5
    MIN_VOCABULARY_HITS = 3

    def __init__(self, model_path: str = None, master_key: bytes = None):
        self.min_confidence = self.MIN_CONFIDENCE
        self.metadata = None

        if model_path and master_key and os.path.exists(model_path):
            try:
                self.model, self.metadata = load_model(model_path, master_key)
                self.min_confidence = self.metadata.get("min_confidence", self.MIN_CONFIDENCE)
                logger.info("Loaded trained classifier (%s; trained %s on %d documents).",
                            ", ".join(self.metadata["labels"]), self.metadata["trained_at"],
                            sum(self.metadata["samples_per_label"].values()))
                return
            except Exception as e:
                logger.error("Refusing to load trained classifier at %s: %s. Falling back to the built-in model.",
                             model_path, e)
        elif model_path:
            logger.warning("No trained classifier found at %s; using the built-in demo model. "
                           "Run `python main.py train` once you have labelled documents.", model_path)

        self.model = build_demo_pipeline()
        self._train_local_model()

    def _train_local_model(self):
        """Trains the built-in fallback model on synthetic SME templates."""
        # Synthetic training data representing typical SME business documents (6 items total)
        training_texts = [
            # Invoices (Samples 1 & 2)
            "INVOICE Bill To: Customer Amount Due KSH 50,000 Supply of office equipment Tax Invoice Bank Details Balance",
            "Invoice invoice payment requested receipt total due bank account number branch swift code billing",

            # Employment Contracts (Samples 3 & 4)
            "EMPLOYMENT CONTRACT This Agreement is made between the Employer and the Employee Job Title Salary Leave Benefits Termination",
            "Employment Agreement probationary period permanent employment position description basic salary allowances ksh",

            # National IDs (Samples 5 & 6)
            "REPUBLIC OF KENYA NATIONAL IDENTITY CARD ID NUMBER ID NO Names Date of Birth Sex Signature copy Holder",
            "KENYA NATIONAL ID CARD Identity Document Date of Issue District of Birth Serial Number card holder copy"
        ]

        # Corresponding targets aligned perfectly (6 items total)
        training_labels = [
            "Invoice", "Invoice",
            "Contract", "Contract",
            "National ID", "National ID"
        ]

        # Trigger training fit
        self.model.fit(training_texts, training_labels)
        logger.info("Built-in demo classification model trained.")

    @property
    def is_trained_model(self) -> bool:
        return self.metadata is not None

    def predict_with_confidence(self, extracted_text: str):
        """Returns (label, confidence). Label is UNKNOWN_LABEL when the model is not confident enough."""
        if not extracted_text.strip():
            return UNKNOWN_LABEL, 0.0

        # How many distinct known keywords the document shares with the training vocabulary
        vocabulary_hits = self.model.named_steps['tfidf'].transform([extracted_text]).nnz
        probabilities = self.model.predict_proba([extracted_text])[0]
        best_index = probabilities.argmax()
        confidence = float(probabilities[best_index])

        if confidence < self.min_confidence or vocabulary_hits < self.MIN_VOCABULARY_HITS:
            return UNKNOWN_LABEL, confidence
        return self.model.classes_[best_index], confidence

    def predict_type(self, extracted_text: str) -> str:
        """Analyzes raw string variables in RAM and outputs the predicted document type."""
        return self.predict_with_confidence(extracted_text)[0]

if __name__ == "__main__":
    classifier = DocumentClassifier()
    test_text = "Tax Invoice details: Total due is KSH 12,000 payable to account standard bank."
    print(f"[*] Test Prediction Result: {classifier.predict_type(test_text)}")
