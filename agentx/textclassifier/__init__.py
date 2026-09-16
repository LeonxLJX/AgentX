# ============================================================================
# AgentX.textclassifier - Module 1: Text Classification
# ============================================================================

"""Text classification for the most frequent freelance outsourcing type.

The module wraps a scikit-learn pipeline (TF-IDF vectorizer + linear model)
behind a small, clean API. It handles English and Chinese text, multi-class
problems, probability output and model persistence.

Typical outsourcing fit
-----------------------
- Comment / ticket / email classification
- Sentiment polarity classification (positive / negative / neutral)
- Intent routing for support desks
- Document type tagging

Quick start
-----------
>>> from agentx.textclassifier import TextClassifier
>>> from agentx.textclassifier.dataset import load_ticket
>>> texts, labels = load_ticket()
>>> clf = TextClassifier().fit(texts, labels)
>>> clf.predict(["When will my package arrive?"])
[ClassificationResult(text='When will my package arrive?', label='logistics', confidence=0.93, ...)]
"""

from agentx.textclassifier.classifier import MODEL_REGISTRY, TextClassifier
from agentx.textclassifier.dataset import load_sentiment, load_ticket

__all__ = ["TextClassifier", "MODEL_REGISTRY", "load_sentiment", "load_ticket"]
