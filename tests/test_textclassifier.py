# ============================================================================
# Tests: TextClassifier
# ============================================================================

import pytest

from agentx.textclassifier import TextClassifier
from agentx.textclassifier.dataset import load_sentiment, load_ticket


def test_fit_and_predict_ticket():
    texts, labels = load_ticket()
    clf = TextClassifier().fit(texts, labels)
    results = clf.predict(["I want to request a refund, the item has not shipped yet."])
    assert results[0].label == "refund"
    assert 0.0 <= results[0].confidence <= 1.0


def test_predict_accepts_single_string():
    texts, labels = load_ticket()
    clf = TextClassifier().fit(texts, labels)
    result = clf.predict("When will my package arrive?")
    assert result.label == "logistics"


def test_evaluate_returns_metrics():
    texts, labels = load_sentiment()
    clf = TextClassifier().fit(texts, labels)
    report = clf.evaluate(texts, labels)
    assert report["accuracy"] >= 0.5
    assert "f1_macro" in report and "report" in report


def test_save_load_roundtrip(tmp_path):
    texts, labels = load_ticket()
    clf = TextClassifier().fit(texts, labels)
    path = tmp_path / "model.joblib"
    clf.save(str(path))
    restored = TextClassifier.load(str(path))
    assert restored.predict(["I forgot my password, how do I recover it?"])[0].label == "account"


def test_fit_requires_same_length():
    with pytest.raises(ValueError):
        TextClassifier().fit(["a", "b"], ["x"])


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        TextClassifier(model="nope")


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TextClassifier().predict(["hello"])
