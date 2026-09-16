# ============================================================================
# AgentX.phasedetect - Module 2: NLP Phase Detection
# ============================================================================

"""NLP phase detection: sentiment, topics, keywords and entity extraction.

A lightweight, dependency-friendly NLP layer that covers the most requested
freelance NLP sub-tasks:

- **Sentiment analysis**  : lexicon-based polarity with negation and
  intensifier handling (English + Chinese).
- **Topic detection**     : TF-IDF + K-means clustering over a document set;
  each cluster is labelled with its top terms.
- **Keyword extraction**  : stop-word-filtered token scoring (character
  bigrams for Chinese, no segmentation dependency required).
- **Entity extraction**   : regex patterns for emails, URLs, phone numbers,
  dates, monetary amounts and numeric values.

Quick start
-----------
>>> from agentx.phasedetect import PhaseDetector
>>> detector = PhaseDetector(language="auto")
>>> result = detector.analyze("The service is extremely good and fast!")
>>> result.sentiment, result.sentiment_score
('positive', 2.6)
"""

from agentx.phasedetect.detector import PhaseDetector

__all__ = ["PhaseDetector"]
