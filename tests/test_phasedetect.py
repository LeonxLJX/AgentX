# ============================================================================
# Tests: PhaseDetect
# ============================================================================

import pytest

from agentx.phasedetect import PhaseDetector


def test_english_sentiment_positive():
    detector = PhaseDetector(language="en")
    label, score = detector.sentiment_of("The service is extremely good and fast!")
    assert label == "positive"
    assert score > 1.0  # intensifier multiplies


def test_chinese_sentiment_negative():
    # The detector ships bilingual lexicons; Chinese is supported too.
    detector = PhaseDetector(language="zh")
    label, score = detector.sentiment_of("产品质量太差了，服务也很糟糕。")
    assert label == "negative"


def test_negation_flips_sentiment():
    detector = PhaseDetector(language="en")
    label, _ = detector.sentiment_of("This is not good at all")
    assert label in {"negative", "neutral"}


def test_entities_extraction():
    detector = PhaseDetector(language="en")
    result = detector.analyze("Mail me at a@b.com or call 138-0000-1234 on 2026-09-20.")
    assert "EMAIL" in result.entities
    assert "PHONE" in result.entities
    assert "DATE" in result.entities


def test_keywords_nonempty():
    detector = PhaseDetector(language="en")
    keywords = detector.keywords("The delivery was fast and the packaging was excellent.")
    assert len(keywords) >= 2


def test_fit_topics_and_predict():
    detector = PhaseDetector(language="en")
    corpus = [
        "How do I get a refund for my order?",
        "I want my money back please.",
        "Where is my package, tracking is stuck.",
        "The delivery is delayed again.",
        "My account is locked and I cannot log in.",
        "Reset my password please.",
    ]
    detector.fit_topics(corpus, n_topics=3)
    cid, terms = detector.predict_topic("The delivery is late again")
    assert 0 <= cid < 3
    assert terms


def test_predict_topic_before_fit_raises():
    detector = PhaseDetector()
    with pytest.raises(RuntimeError):
        detector.predict_topic("hello")


def test_empty_text_neutral():
    detector = PhaseDetector()
    result = detector.analyze("")
    assert result.sentiment == "neutral"
