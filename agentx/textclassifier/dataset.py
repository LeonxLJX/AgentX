# ============================================================================
# AgentX.textclassifier.dataset - built-in demo datasets
# ============================================================================

"""Small hand-curated datasets so the module is demonstrable out of the box.

Two datasets are provided:

- ``load_sentiment`` : 24 English movie-review snippets (positive / negative).
- ``load_ticket``    : 36 English support tickets across 4 intents
  (``refund`` / ``logistics`` / ``account`` / ``tech``).

Each loader returns ``(texts, labels)`` ready for ``TextClassifier.fit``.
"""

from __future__ import annotations

from typing import List, Tuple

# --------------------------------------------------------------------------
# English sentiment (positive / negative)
# --------------------------------------------------------------------------
_SENTIMENT_POSITIVE = [
    "Absolutely loved this film, the acting was brilliant.",
    "A masterpiece - great story and wonderful cinematography.",
    "Funny, heartwarming and beautifully directed.",
    "The best movie I have seen this year, highly recommended.",
    "Amazing performances and a touching ending.",
    "Great pacing, clever dialogue and a satisfying plot.",
    "An emotional rollercoaster that stayed with me for days.",
    "Superb soundtrack and stunning visuals, a must watch.",
    "Charming, witty and full of heart.",
    "Brilliantly crafted with fantastic character development.",
    "I enjoyed every minute, absolutely delightful.",
    "A triumph of modern cinema, brilliant from start to finish.",
]

_SENTIMENT_NEGATIVE = [
    "Boring and predictable, I almost fell asleep.",
    "Terrible acting and a nonsensical plot.",
    "Waste of time, the story goes nowhere.",
    "Dreadful pacing, the movie drags on forever.",
    "Poor direction and awful dialogue.",
    "I hated every second of it, deeply disappointing.",
    "Mediocre at best, cliched characters and weak writing.",
    "A dull mess of random scenes with no purpose.",
    "Confusing, overlong and ultimately forgettable.",
    "Bad editing ruined what could have been a decent film.",
    "Uninspired and lifeless, skip it.",
    "Frustratingly shallow with a laughable ending.",
]

# --------------------------------------------------------------------------
# English support tickets (4 intents)
# --------------------------------------------------------------------------
_TICKET_REFUND = [
    "I want to request a refund, the item has not shipped yet.",
    "How do I apply for a return and refund on my order?",
    "Customer service, I would like to return this item.",
    "The item I received has quality issues, please refund me.",
    "I changed my mind after paying, can I get a full refund?",
    "When will my refund arrive? I applied three days ago.",
    "The item does not match the description, I want a return and refund.",
    "How do I cancel my refund request?",
    "The order shows refunded but the money has not arrived.",
]

_TICKET_LOGISTICS = [
    "When will my package arrive?",
    "Shipping info has not updated for days.",
    "The parcel shows delivered but I did not receive it.",
    "Can you check where my package is right now?",
    "Delivery is delayed again, when will it arrive?",
    "I entered the wrong shipping address, can I change it?",
    "Tracking has shown in transit forever, is that normal?",
    "My package is lost, what should I do?",
    "How long does same-city delivery usually take?",
]

_TICKET_ACCOUNT = [
    "I forgot my password, how do I recover it?",
    "My account is frozen, how do I unlock it?",
    "I changed my phone number, how do I update it?",
    "How do I permanently delete my account?",
    "Login keeps saying the verification code is wrong.",
    "My account was stolen, urgent help needed.",
    "How do I change the email on my account?",
    "Real-name verification keeps failing, what should I do?",
    "Can I transfer my account to someone else?",
]

_TICKET_TECH = [
    "The app crashes on launch, how do I fix it?",
    "The payment page keeps loading forever.",
    "The program will not start after installation.",
    "The interface looks broken after the update.",
    "Video playback keeps stuttering.",
    "QR-code login says network error.",
    "Data sync keeps failing, what is the cause?",
    "My printer will not connect and the driver will not install.",
    "The system says out of memory, how do I clean it?",
]


def _pair(positive: List[str], negative: List[str]) -> Tuple[List[str], List[str]]:
    texts = positive + negative
    labels = ["positive"] * len(positive) + ["negative"] * len(negative)
    return texts, labels


def load_sentiment() -> Tuple[List[str], List[str]]:
    """Return English movie-review sentiment data ``(texts, labels)``."""
    return _pair(_SENTIMENT_POSITIVE, _SENTIMENT_NEGATIVE)


def load_ticket() -> Tuple[List[str], List[str]]:
    """Return English support-ticket intent data ``(texts, labels)``."""
    groups = [
        (_TICKET_REFUND, "refund"),
        (_TICKET_LOGISTICS, "logistics"),
        (_TICKET_ACCOUNT, "account"),
        (_TICKET_TECH, "tech"),
    ]
    texts: List[str] = []
    labels: List[str] = []
    for group_texts, label in groups:
        texts.extend(group_texts)
        labels.extend([label] * len(group_texts))
    return texts, labels
