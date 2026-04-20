"""
features.py — Stage B: Context-Aware Feature Extraction
Maps AnnotatedSentences to numeric feature vectors. Context downweights hits:
e.g. hits in quotes are excluded, negated hits are 0.2x.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Tuple
from biaslab.config import (SENSATIONAL_WORDS, ABSOLUTE_WORDS, HEDGE_WORDS, POSITIVE_WORDS, NEGATIVE_WORDS, 
                            LEFT_LOADED, RIGHT_LOADED, FRAMING_WORDS, SUPERLATIVES, CITATION_SIGNALS, 
                            WEASEL_SIGNALS, WEIGHT_AUTHOR_VOICE, WEIGHT_ATTRIBUTED, WEIGHT_QUOTED, WEIGHT_NEGATED, NEGATION_WINDOW)
from biaslab.preprocessing import AnnotatedSentence

@dataclass
class SentenceFeatures:
    """Feature vector for a single sentence."""
    sentence_index: int
    word_count: int
    is_headline: bool
    is_quoted: bool
    is_attributed: bool
    context_weight: float
    
    # Extracted stats
    exclamation_count: int = 0
    caps_word_count: int = 0
    n_negated_hits: int = 0
    negated_phrases: List[str] = field(default_factory=list)
    excluded_quoted: List[str] = field(default_factory=list)

    # Dictionary storing raw hit count and found phrases per lexicon axis
    axis_data: dict = field(default_factory=lambda: {
        "sensational": {"raw": 0, "found": []},
        "positive": {"raw": 0, "found": []},
        "negative": {"raw": 0, "found": []},
        "absolute": {"raw": 0, "found": []},
        "hedge": {"raw": 0, "found": []},
        "weasel": {"raw": 0, "found": []},
        "citation": {"raw": 0, "found": []},
        "framing": {"raw": 0, "found": []},
        "superlative": {"raw": 0, "found": []},
        "left": {"raw": 0, "found": []},
        "right": {"raw": 0, "found": []},
    })

def _is_negated(words: List[str], phrase_words: List[str], neg_positions: List[int]) -> bool:
    """Check if phrase occurs within NEGATION_WINDOW tokens after a negation."""
    if not phrase_words or not neg_positions: return False
    pw_len = len(phrase_words)
    for i in range(len(words) - pw_len + 1):
        if words[i:i + pw_len] == phrase_words:
            if any(0 < (i - np) <= NEGATION_WINDOW for np in neg_positions): return True
    return False

def _count_in_context(sentence: AnnotatedSentence, phrases: List[str]) -> Tuple[int, List[str], int, List[str], List[str]]:
    """Counts phrase occurrences, separating valid hits, negated hits, and quoted exclusions."""
    valid, negated_c, found, negated_p, excluded_q = 0, 0, [], [], []
    
    # Exclude all if sentence is fully quoted
    if sentence.is_quoted:
        excluded_q = [p for p in phrases if re.search(r"\b" + re.escape(p) + r"\b", sentence.text_lower)]
        return 0, [], 0, [], excluded_q
        
    for p in phrases:
        hits = len(re.findall(r"\b" + re.escape(p) + r"\b", sentence.text_lower))
        if hits == 0: continue
        
        # Check quoted partial spans (via case-preserved text)
        is_quoted_hit = any(any(s <= m.start() < e for s,e in sentence.quoted_spans) 
                            for m in re.finditer(r"\b" + re.escape(p) + r"\b", sentence.text, re.IGNORECASE))
        if is_quoted_hit:
            excluded_q.append(p)
            continue
            
        if _is_negated(sentence.words, p.lower().split(), sentence.negation_positions):
            negated_c += hits; negated_p.append(p)
        else:
            valid += hits; found.append(p)
            
    return valid, sorted(set(found)), negated_c, negated_p, excluded_q

def extract_sentence_features(s: AnnotatedSentence) -> SentenceFeatures:
    """Computes all context-aware features for a pre-annotated sentence."""
    wt = WEIGHT_QUOTED if s.is_quoted else WEIGHT_ATTRIBUTED if s.is_attributed else WEIGHT_AUTHOR_VOICE
    feats = SentenceFeatures(s.index, s.word_count, s.is_headline, s.is_quoted, s.is_attributed, wt,
                             exclamation_count=s.text.count("!"),
                             caps_word_count=len(re.findall(r"\b[A-Z]{4,}\b", s.text)))

    # Feature mapping definitions
    mappings = {
        "sensational": SENSATIONAL_WORDS, "positive": POSITIVE_WORDS, "negative": NEGATIVE_WORDS,
        "absolute": ABSOLUTE_WORDS, "hedge": HEDGE_WORDS, "weasel": WEASEL_SIGNALS,
        "citation": CITATION_SIGNALS, "framing": FRAMING_WORDS, "superlative": SUPERLATIVES,
        "left": LEFT_LOADED, "right": RIGHT_LOADED
    }
    
    for axis, vocab in mappings.items():
        v_count, v_found, n_count, n_found, exc_q = _count_in_context(s, vocab)
        feats.axis_data[axis]["raw"] = v_count
        feats.axis_data[axis]["found"] = v_found
        feats.n_negated_hits += n_count
        feats.negated_phrases.extend(n_found)
        feats.excluded_quoted.extend(exc_q)

    return feats

def extract_all_features(sentences: List[AnnotatedSentence]) -> List[SentenceFeatures]:
    return [extract_sentence_features(s) for s in sentences]
