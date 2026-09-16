# ============================================================================
# Example: TextClassifier - support-ticket intent classification
# Run:  python examples/demo_textclassifier.py
# ============================================================================

"""Demonstrates training, predicting, evaluating and persisting a classifier."""

from agentx.core import get_logger
from agentx.textclassifier import TextClassifier
from agentx.textclassifier.dataset import load_sentiment, load_ticket

log = get_logger("demo.textclassifier")

if __name__ == "__main__":
    # --- 1. Support-ticket intent classification ----------------------------
    texts, labels = load_ticket()
    clf = TextClassifier(ngram_range=(1, 2)).fit(texts, labels)

    probes = [
        "I want to request a refund, the item has not shipped yet.",
        "Delivery is delayed again, when will it arrive?",
        "The app crashes on launch, how do I fix it?",
    ]
    for result in clf.predict(probes):
        print(f"[{result.label:>9} {result.confidence:.2%}] {result.text}")
        print(f"    scores={ {k: round(v, 3) for k, v in result.scores.items()} }")

    # --- 2. Evaluation + persistence ----------------------------------------
    eval_report = clf.evaluate(texts, labels)
    print(f"\naccuracy={eval_report['accuracy']:.3f}  f1_macro={eval_report['f1_macro']:.3f}")

    clf.save("outputs/ticket_model.joblib")
    restored = TextClassifier.load("outputs/ticket_model.joblib")
    print("restored prediction:", restored.predict(["I forgot my password, how do I recover it?"])[0].label)

    # --- 3. English sentiment (binary) --------------------------------------
    en_texts, en_labels = load_sentiment()
    en_clf = TextClassifier().fit(en_texts, en_labels)
    r = en_clf.predict(["This movie is an absolute masterpiece!"])[0]
    print(f"\nen sentiment: {r.label} ({r.confidence:.2%})")
