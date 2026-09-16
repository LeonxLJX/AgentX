# ============================================================================
# AgentX.phasedetect.detector - PhaseDetector implementation
# ============================================================================

"""Phase detection over a single document or a whole corpus.

The detector is fully rule + scikit-learn based, so it runs anywhere without
heavy model downloads, yet covers the phases clients actually ask for:
sentiment, topics, keywords and entities.

Language handling
-----------------
- ``language="auto"`` : heuristics decide between English and Chinese based
  on the presence of CJK characters.
- ``language="en"`` / ``"zh"`` : force a language.
- Chinese tokenization uses stop-word-filtered character bigrams, so no
  external segmentation library is required (jieba can be plugged in by
  replacing :meth:`PhaseDetector._tokenize`).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from agentx.core import get_logger
from agentx.core.schema import NlpPhase

logger = get_logger("agentx.phasedetect")

# ============================================================================
# Lexicons
# ============================================================================
EN_POSITIVE = {
    "good", "great", "excellent", "amazing", "love", "loved", "wonderful",
    "brilliant", "fantastic", "happy", "nice", "best", "superb", "delightful",
    "enjoy", "enjoyed", "masterpiece", "charming", "witty", "heartwarming",
    "touching", "stunning", "helpful", "fast", "reliable", "recommend",
    "perfect", "impressive", "awesome", "beautiful", "friendly", "satisfied",
}
EN_NEGATIVE = {
    "bad", "terrible", "awful", "boring", "poor", "worst", "hate", "hated",
    "disappointing", "dreadful", "dull", "confusing", "frustrating",
    "shallow", "laughable", "mediocre", "nonsensical", "lifeless",
    "uninspired", "waste", "fail", "failed", "slow", "stupid", "broken",
    "useless", "expensive", "rude", "unhappy", "annoying", "buggy",
}
ZH_POSITIVE = {
    "好", "棒", "赞", "喜欢", "爱", "满意", "优质", "完美", "优秀", "出色",
    "推荐", "开心", "好评", "快", "方便", "实用", "惊喜", "感谢", "靠谱",
    "专业", "流畅", "稳定", "清晰", "实惠", "贴心", "热情",
}
ZH_NEGATIVE = {
    "差", "坏", "烂", "垃圾", "慢", "贵", "失望", "难用", "问题", "错误",
    "故障", "卡", "闪退", "失败", "投诉", "不满", "亏", "假", "坑", "烦",
    "崩溃", "延迟", "丢失", "异常", "糟糕",
}
NEGATORS_EN = {
    "not", "no", "never", "none", "hardly", "barely", "don't", "doesnt",
    "didn't", "cant", "cannot", "won't", "isn't", "wasn't", "aren't",
}
NEGATORS_ZH = {"不", "没", "无", "别", "非", "未"}
INTENSIFIERS_EN = {
    "very": 1.5, "extremely": 2.0, "really": 1.3, "so": 1.4,
    "absolutely": 1.6, "totally": 1.5, "incredibly": 1.8, "quite": 1.2,
    "too": 1.3, "super": 1.6,
}
INTENSIFIERS_ZH = {
    "非常": 1.6, "太": 1.5, "很": 1.3, "特别": 1.5, "十分": 1.4,
    "极其": 2.0, "超": 1.6, "超级": 1.7, "格外": 1.5,
}

EN_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "for", "on",
    "with", "is", "are", "was", "were", "be", "been", "it", "this", "that",
    "i", "you", "he", "she", "we", "they", "my", "your", "at", "as", "by",
    "from", "have", "has", "had", "do", "does", "did", "will", "would",
    # NOTE: negators ("not", "no", ...) are deliberately NOT in the stop
    # list so negation flipping keeps working in sentiment scoring.
}
ZH_STOP_CHARS = {
    "的", "了", "是", "我", "你", "他", "她", "它", "们", "在", "有", "就",
    "都", "而", "及", "与", "着", "或", "一个", "和", "也", "很", "这个",
    "那个", "吗", "呢", "啊", "吧", "哦", "么",
}

# Entity regex patterns (ordered by specificity).
_PATTERNS: Dict[str, str] = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "URL": r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    "PHONE": r"(?:\+?\d{1,3}[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}",
    "DATE": r"\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?",
    "MONEY": r"[$¥￥€£]\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:元|美元|欧元|英镑)",
    "NUMBER": r"\b\d+(?:\.\d+)?\b",
}


class PhaseDetector:
    """Detects sentiment, topics, keywords and entities in text.

    Parameters
    ----------
    language : {"auto", "en", "zh"}
        Default language for tokenization and sentiment lexicons.
    min_keyword_len : int
        Minimum token length for keyword extraction.
    top_k : int
        Default number of keywords / topics returned.
    """

    def __init__(
        self,
        language: str = "auto",
        min_keyword_len: int = 2,
        top_k: int = 5,
    ) -> None:
        if language not in {"auto", "en", "zh"}:
            raise ValueError(f"language must be one of auto/en/zh, got {language}")
        self.language = language
        self.min_keyword_len = min_keyword_len
        self.top_k = top_k
        self.topic_model_: Optional[Any] = None
        self.topic_vectorizer_: Optional[TfidfVectorizer] = None
        self.topic_terms_: List[List[str]] = []
        self._compiled: Dict[str, re.Pattern] = {
            key: re.compile(pat) for key, pat in _PATTERNS.items()
        }

    # ============================================================== public
    def analyze(self, text: str) -> NlpPhase:
        """Run the full detection pipeline over one document.

        Returns
        -------
        NlpPhase
            Sentiment label/score, extracted keywords and entities. Topics are
            appended only when a topic model has been fitted via
            :meth:`fit_topics`.
        """
        if not text or not text.strip():
            return NlpPhase(text=text or "", sentiment="neutral", sentiment_score=0.0)

        label, score = self.sentiment_of(text)
        keywords = self.keywords(text, top_k=self.top_k)
        entities = self.entities(text)
        topics: List[str] = []
        if self.topic_model_ is not None:
            topic_id, terms = self.predict_topic(text)
            topics = [f"topic_{topic_id}"] + terms[: self.top_k]

        return NlpPhase(
            text=text,
            sentiment=label,
            sentiment_score=score,
            topics=topics,
            entities=entities,
            keywords=keywords,
        )

    # ------------------------------------------------------------- sentiment
    def sentiment_of(self, text: str) -> Tuple[str, float]:
        """Lexicon-based sentiment scoring with negation & intensifiers.

        Returns
        -------
        tuple[str, float]
            ``("positive" | "neutral" | "negative", score)``.
        """
        lang = self._detect_lang(text)
        tokens = self._tokenize(text, lang)
        pos, neg = self._lexicons(lang)

        score = 0.0
        negate = False
        for token in tokens:
            if token in NEGATORS_EN or token in NEGATORS_ZH:
                negate = True
                continue
            intensity = 1.0
            if token in INTENSIFIERS_EN:
                intensity = INTENSIFIERS_EN[token]
                continue
            if token in INTENSIFIERS_ZH:
                intensity = INTENSIFIERS_ZH[token]
                continue

            if token in pos:
                score += intensity if not negate else -intensity
            elif token in neg:
                score -= intensity if not negate else -intensity
            negate = False

        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        return label, round(score, 3)

    # ------------------------------------------------------------- keywords
    def keywords(self, text: str, top_k: Optional[int] = None) -> List[str]:
        """Extract the most salient tokens (stop words removed)."""
        lang = self._detect_lang(text)
        tokens = self._tokenize(text, lang)
        tokens = [t for t in tokens if len(t) >= self.min_keyword_len]
        counts = Counter(tokens)
        k = top_k or self.top_k
        return [tok for tok, _ in counts.most_common(k)]

    # ------------------------------------------------------------- entities
    def entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities via regex patterns, grouped by type."""
        found: Dict[str, List[str]] = {}
        for name, pattern in self._compiled.items():
            matches = [m.group(0) for m in pattern.finditer(text)]
            # Deduplicate while preserving order.
            seen: List[str] = []
            for m in matches:
                if m not in seen:
                    seen.append(m)
            if seen:
                found[name] = seen
        return found

    # ------------------------------------------------------------- topics
    def fit_topics(self, texts: List[str], n_topics: int = 3) -> "PhaseDetector":
        """Cluster a document set into topics (TF-IDF + K-means).

        Stores a fitted model; subsequent :meth:`analyze` calls attach the
        predicted topic and its top terms to every document.
        """
        texts = [t for t in texts if t and t.strip()]
        if len(texts) < 2:
            raise ValueError("fit_topics needs at least 2 non-empty documents")
        n_topics = max(1, min(n_topics, len(texts)))

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b|[^\W\d_]{2}",
        )
        matrix = vectorizer.fit_transform(texts)

        kmeans: Optional[KMeans] = None
        if n_topics == 1:
            assignments = np.zeros(len(texts), dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_topics, n_init=10, random_state=42)
            assignments = kmeans.fit_predict(matrix)

        terms = vectorizer.get_feature_names_out()
        self.topic_terms_ = []
        for cid in range(n_topics):
            mask = assignments == cid
            if mask.sum() == 0:
                self.topic_terms_.append([])
                continue
            centroid = np.asarray(matrix[mask].sum(axis=0)).ravel()
            top_idx = np.argsort(centroid)[::-1][: self.top_k]
            self.topic_terms_.append([str(terms[i]) for i in top_idx])

        # Keep the fitted KMeans object so predict_topic is exact and simple.
        self.topic_model_ = {
            "n_topics": n_topics,
            "kmeans": kmeans,
        }
        self.topic_vectorizer_ = vectorizer
        logger.info("fitted %d topics over %d documents", n_topics, len(texts))
        return self

    def predict_topic(self, text: str) -> Tuple[int, List[str]]:
        """Assign a document to a fitted topic cluster.

        Returns
        -------
        tuple[int, list[str]]
            Cluster id and its top representative terms.
        """
        if self.topic_model_ is None or self.topic_vectorizer_ is None:
            raise RuntimeError("call fit_topics() before predict_topic()")
        X = self.topic_vectorizer_.transform([text])
        kmeans = self.topic_model_["kmeans"]
        cid = int(kmeans.predict(X)[0]) if kmeans is not None else 0
        return cid, self.topic_terms_[cid]

    # ============================================================== helpers
    def _detect_lang(self, text: str) -> str:
        if self.language != "auto":
            return self.language
        return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"

    @staticmethod
    def _lexicons(lang: str) -> Tuple[set, set]:
        if lang == "zh":
            return ZH_POSITIVE, ZH_NEGATIVE
        return EN_POSITIVE, EN_NEGATIVE

    def _tokenize(self, text: str, lang: str) -> List[str]:
        """Language-aware tokenization.

        - English: lowercase words, stop words removed.
        - Chinese: stop-character-filtered bigrams (no segmentation lib).
        """
        if lang == "zh":
            chars = [
                c for c in text if not c.isspace() and c not in ZH_STOP_CHARS
            ]
            bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
            # Keep bigrams that contain at least one meaningful char.
            return [b for b in bigrams if any(c not in ZH_STOP_CHARS for c in b)]
        words = re.findall(r"[a-zA-Z']+", text.lower())
        return [w for w in words if w not in EN_STOP_WORDS]
