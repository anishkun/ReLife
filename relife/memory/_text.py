"""Shared tokenization for keyword recall (store + skills).

Drops stopwords so common filler words ("a", "the", "to", "please") don't create
spurious matches between unrelated queries and memories.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "it", "be", "with", "my", "me", "i", "you", "your", "this", "that", "do",
    "did", "please", "want", "need", "new", "use", "using", "can", "should",
    "would", "could", "will", "just", "from", "at", "as", "by", "so", "if",
    "then", "but", "not", "no", "yes", "all", "any", "how", "what", "when",
    "where", "why", "make", "made", "get", "got", "set", "up",
}


def tokenize(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOPWORDS}
