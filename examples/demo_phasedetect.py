# ============================================================================
# Example: PhaseDetect - sentiment / topics / entities
# Run:  python examples/demo_phasedetect.py
# ============================================================================

"""Demonstrates full NLP phase detection over single documents and corpora.

The detector is bilingual (English + Chinese lexicons); English is used in
this demo for readability.
"""

from agentx.phasedetect import PhaseDetector

if __name__ == "__main__":
    detector = PhaseDetector(language="auto")

    # --- Single-document analysis -------------------------------------------
    samples = [
        "The service is extremely good and fast, contact support@example.com.",
        "The product quality is terrible, delivery is three days late and "
        "support at 138-0000-1234 never answers.",
        "The refund of 299.00 USD for order #8841 should arrive by 2026-09-20.",
    ]
    for text in samples:
        result = detector.analyze(text)
        print(f"\nTEXT: {text}")
        print(f"  sentiment : {result.sentiment} ({result.sentiment_score:+.2f})")
        print(f"  keywords  : {result.keywords}")
        print(f"  entities  : {result.entities}")

    # --- Corpus-level topic clustering ---------------------------------------
    corpus = [
        "How do I get a refund for my order?",
        "I want my money back please.",
        "Where is my package, tracking is stuck.",
        "The delivery is delayed again.",
        "My account is locked and I cannot log in.",
        "Reset my password please.",
    ]
    detector.fit_topics(corpus, n_topics=3)
    for text in corpus:
        cid, terms = detector.predict_topic(text)
        print(f"\n[cluster {cid}] {text}\n    top terms: {terms}")
