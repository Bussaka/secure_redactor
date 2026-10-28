import logging
import os
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.classifier import MIN_DOCUMENT_FREQUENCY, build_pipeline, save_model

logger = logging.getLogger(__name__)

MIN_DOCUMENTS_PER_LABEL = 2
RECOMMENDED_DOCUMENTS_PER_LABEL = 10

class TrainingDataError(ValueError):
    """The training folder does not contain enough usable labelled documents."""

def load_training_documents(training_dir: str, extractor, redactor):
    """Reads data/training/<Label>/<files>. Returns (texts, labels, skipped_files).

    Text is redacted BEFORE training, so names and other PII never end up in the saved model's vocabulary.
    """
    texts, labels, skipped = [], [], []
    if not os.path.isdir(training_dir):
        raise TrainingDataError(f"Training folder not found: {training_dir}")

    for label in sorted(os.listdir(training_dir)):
        label_dir = os.path.join(training_dir, label)
        if label.startswith('.') or not os.path.isdir(label_dir):
            continue
        for file_name in sorted(os.listdir(label_dir)):
            path = os.path.join(label_dir, file_name)
            if file_name.startswith('.') or not os.path.isfile(path):
                continue
            text = extractor.process_file(path)
            if not text.strip():
                skipped.append(os.path.join(label, file_name))
                continue
            texts.append(redactor.redact_text(text))
            labels.append(label)
    return texts, labels, skipped

def validate_training_set(labels):
    counts = Counter(labels)
    if len(counts) < 2:
        raise TrainingDataError(
            f"Need at least 2 document categories (sub-folders) with readable documents, found {len(counts)}.")
    too_small = {label: n for label, n in counts.items() if n < MIN_DOCUMENTS_PER_LABEL}
    if too_small:
        raise TrainingDataError(
            f"Each category needs at least {MIN_DOCUMENTS_PER_LABEL} readable documents; too few in: {too_small}")
    for label, n in sorted(counts.items()):
        if n < RECOMMENDED_DOCUMENTS_PER_LABEL:
            logger.warning("Category '%s' has only %d documents; %d+ is recommended for reliable predictions.",
                           label, n, RECOMMENDED_DOCUMENTS_PER_LABEL)
    return counts

def evaluate(texts, labels, min_confidence: float) -> dict:
    """Estimates real-world performance with stratified cross-validation (each document is scored by a model
    that never saw it during training)."""
    n_splits = min(5, min(Counter(labels).values()))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probabilities = cross_val_predict(build_pipeline(), texts, labels, cv=folds, method="predict_proba")

    classes = np.array(sorted(set(labels)))
    predicted = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    truth = np.array(labels)
    confident = confidence >= min_confidence

    return {
        "folds": n_splits,
        "accuracy": round(float((predicted == truth).mean()), 4),
        # Share of documents that would get a real label rather than "Unknown" at the chosen threshold
        "coverage_at_threshold": round(float(confident.mean()), 4),
        "accuracy_when_confident": round(float((predicted[confident] == truth[confident]).mean()), 4)
                                   if confident.any() else None,
        "report": classification_report(truth, predicted, zero_division=0),
    }

def train_classifier(training_dir: str, model_path: str, master_key: bytes, extractor, redactor,
                     min_confidence: float = 0.5) -> dict:
    """Full training run: load + redact documents, cross-validate, fit on everything, save a signed model."""
    texts, labels, skipped = load_training_documents(training_dir, extractor, redactor)
    for path in skipped:
        logger.warning("Skipped unreadable training document: %s", path)
    counts = validate_training_set(labels)

    logger.info("Training on %d documents across %d categories...", len(labels), len(counts))
    try:
        evaluation = evaluate(texts, labels, min_confidence)
        model = build_pipeline()
        model.fit(texts, labels)
    except ValueError as e:
        # Raised by the vectorizer when no word appears in enough documents to be kept
        if "no terms remain" not in str(e) and "empty vocabulary" not in str(e):
            raise
        raise TrainingDataError(
            f"Not enough shared wording across documents to learn from (a word must appear in at least "
            f"{MIN_DOCUMENT_FREQUENCY} documents). Add more examples per category.") from e
    metadata = save_model(model, model_path, master_key, {
        "labels": sorted(counts),
        "samples_per_label": dict(sorted(counts.items())),
        "skipped_documents": len(skipped),
        "min_confidence": min_confidence,
        "evaluation": {k: v for k, v in evaluation.items() if k != "report"},
    })
    metadata["evaluation_report"] = evaluation["report"]
    return metadata
