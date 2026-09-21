# ============================================================================
# AgentX.textclassifier.classifier - TextClassifier main implementation
# ============================================================================

"""Production-oriented text classifier with a scikit-learn pipeline.

Design notes
------------
- **Vectorizer** : ``TfidfVectorizer`` with configurable n-gram range and
  feature cap. Chinese text is tokenized as character n-grams, which works
  without any Chinese segmentation dependency.
- **Model** : logistic regression by default (fast, interpretable, provides
  well-behaved probabilities); ``"svm"`` (LinearSVC) is available for large
  sparse problems. Any scikit-learn estimator can be injected.
- **Output** : every prediction is a :class:`ClassificationResult` carrying
  the label, the confidence and the full score distribution.
- **Persistence** : ``save`` / ``load`` use joblib, so a trained model can be
  shipped as a single artifact to a client.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from agentx.core import get_logger
from agentx.core.exceptions import (
    ModelNotFittedError,
    PersistenceError,
    TextClassificationError,
    ValidationError,
)
from agentx.core.schema import ClassificationResult

logger = get_logger("agentx.textclassifier")

# Registry of built-in estimators (any sklearn estimator can be injected too).
MODEL_REGISTRY: Dict[str, Any] = {
    "logistic": LogisticRegression,
    "svm": LinearSVC,
}


class TextClassifier:
    """A complete text-classification solution.

    Parameters
    ----------
    model : str or sklearn estimator
        ``"logistic"`` (default), ``"svm"``, or any estimator exposing
        ``fit`` / ``predict`` (and ideally ``predict_proba``).
    ngram_range : tuple[int, int]
        Character / word n-gram range for the TF-IDF vectorizer.
    max_features : int
        Maximum vocabulary size (caps memory on huge corpora).
    stop_words : str, list or None
        Passed to the vectorizer (``"english"``, a custom list, or ``None``).
    random_state : int
        Seed for reproducible training.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Fitted ``TfidfVectorizer`` + estimator.
    labels_ : list[str]
        Class names in the order used by the estimator.
    """

    def __init__(
        self,
        model: Union[str, Any] = "logistic",
        ngram_range: tuple = (1, 2),
        max_features: int = 20000,
        stop_words: Union[str, List[str], None] = None,
        random_state: int = 42,
    ) -> None:
        if isinstance(model, str):
            if model not in MODEL_REGISTRY:
                raise TextClassificationError(
                    f"Unknown model '{model}'. Choose from {list(MODEL_REGISTRY)}."
                )
            estimator: Any = MODEL_REGISTRY[model](
                random_state=random_state, max_iter=2000, class_weight="balanced"
            )
        else:
            estimator = model

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
            sublinear_tf=True,
        )
        self.estimator = estimator
        self.pipeline_: Optional[Pipeline] = None
        self.labels_: List[str] = []
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, texts: List[str], labels: List[str]) -> "TextClassifier":
        """Train the pipeline on raw texts and their labels.

        Parameters
        ----------
        texts : list[str]
            Raw documents (``pd.Series`` is accepted too).
        labels : list[str]
            Ground-truth class for each document.

        Returns
        -------
        TextClassifier
            Self, for chaining.
        """
        texts = list(texts)
        labels = list(labels)
        if len(texts) != len(labels):
            raise ValidationError("texts and labels must have the same length")
        if not texts:
            raise ValidationError("cannot fit on an empty dataset")

        self.labels_ = sorted(set(labels))
        logger.info("fitting on %d samples, %d classes", len(texts), len(self.labels_))
        self.pipeline_ = Pipeline(
            steps=[
                ("tfidf", self.vectorizer),
                ("clf", self.estimator),
            ]
        )
        self.pipeline_.fit(texts, labels)
        logger.info("fit complete (vocabulary=%d)", len(self.vectorizer.vocabulary_))
        return self

    # ------------------------------------------------------------- predict
    def _score_map(self, texts: List[str]) -> List[Dict[str, float]]:
        """Return a {label: score} dict per document.

        Uses ``predict_proba`` when available (logistic regression); otherwise
        falls back to ``decision_function`` normalized with a softmax so that
        confidence values are still comparable.
        """
        pipe = self._require_fitted()
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(texts)
        else:
            raw = pipe.decision_function(texts)
            if raw.ndim == 1:  # binary estimator -> two columns
                raw = np.column_stack([-raw, raw])
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)

        return [
            {label: float(p) for label, p in zip(self.labels_, row)}
            for row in proba
        ]

    def predict(self, texts: Union[str, List[str]]) -> List[ClassificationResult]:
        """Predict classes for one or more documents.

        Returns
        -------
        list[ClassificationResult]
            One result per input document.
        """
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        pipe = self._require_fitted()
        labels = list(pipe.predict(batch))
        score_maps = self._score_map(batch)
        results = [
            ClassificationResult(
                text=doc,
                label=lab,
                confidence=float(scores[lab]),
                scores=scores,
            )
            for doc, lab, scores in zip(batch, labels, score_maps)
        ]
        return results[0] if single else results

    # ------------------------------------------------------------ evaluate
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Evaluate accuracy, macro-F1 and a per-class report.

        Returns
        -------
        dict
            ``accuracy``, ``f1_macro``, ``report`` (sklearn classification
            report as dict) and ``n_samples``.
        """
        pipe = self._require_fitted()
        preds = list(pipe.predict(list(texts)))
        report = classification_report(
            list(labels), preds, output_dict=True, zero_division=0
        )
        return {
            "accuracy": accuracy_score(list(labels), preds),
            "f1_macro": f1_score(list(labels), preds, average="macro", zero_division=0),
            "report": report,
            "n_samples": len(list(texts)),
        }

    # --------------------------------------------------------- persistence
    def save(self, path: str) -> str:
        """Persist the fitted model to disk as a joblib artifact.

        Raises
        ------
        PersistenceError
            When the artifact cannot be written.
        """
        self._require_fitted()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            joblib.dump({"pipeline": self.pipeline_, "labels": self.labels_}, path)
        except OSError as exc:
            raise PersistenceError(f"failed to write model to {path}: {exc}") from exc
        logger.info("model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> "TextClassifier":
        """Load a model previously saved with :meth:`save`.

        Raises
        ------
        PersistenceError
            When the artifact cannot be read or is malformed.
        """
        try:
            payload = joblib.load(path)
        except (OSError, KeyError, EOFError) as exc:
            raise PersistenceError(f"failed to load model from {path}: {exc}") from exc
        if not isinstance(payload, dict) or "pipeline" not in payload or "labels" not in payload:
            raise PersistenceError(f"malformed model artifact at {path}")
        obj = cls()
        obj.pipeline_ = payload["pipeline"]
        obj.labels_ = payload["labels"]
        logger.info("model loaded from %s", path)
        return obj

    # ------------------------------------------------------------- helpers
    def _require_fitted(self) -> Pipeline:
        if self.pipeline_ is None:
            raise ModelNotFittedError("model is not fitted - call .fit(texts, labels) first")
        return self.pipeline_

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "fitted" if self.pipeline_ is not None else "not fitted"
        return f"<TextClassifier model={type(self.estimator).__name__} {state}>"
