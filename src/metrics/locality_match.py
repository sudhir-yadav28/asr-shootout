"""Locality Hit Rate — the business metric.

A clip 'passes' if the model captured the locality entity, even with imperfect spelling.
Strategy:
1. Build candidate list: canonical locality + all aliases (Roman + Devanagari).
2. For each candidate, slide a window over the hypothesis and take the best fuzzy match
   (rapidfuzz partial_ratio normalised to 0-100).
3. Hit = score >= HIT_THRESHOLD on any candidate.
   Partial = PARTIAL_THRESHOLD <= score < HIT_THRESHOLD.
   Miss = below PARTIAL_THRESHOLD.

The match is script-aware: if the hypothesis is Devanagari, we still test against Roman
aliases via case+punctuation normalisation, but the Devanagari aliases will naturally win.
"""
from dataclasses import dataclass
import re
import unicodedata
from rapidfuzz import fuzz


HIT_THRESHOLD = 85
PARTIAL_THRESHOLD = 60


def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"[।,.!?;:\"'()\[\]{}\-–—]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@dataclass
class LocalityScore:
    score: float            # 0-100, best fuzzy match across all aliases
    matched_alias: str      # which alias produced the best score
    matched_text: str       # the text span in the hypothesis that matched
    hit: bool
    partial: bool


def score_locality(hyp: str, locality: str, aliases: list[str]) -> LocalityScore:
    candidates = [locality] + (aliases or [])
    hyp_n = _norm(hyp)
    if not hyp_n:
        return LocalityScore(0.0, candidates[0], "", False, False)

    best_score = 0.0
    best_alias = candidates[0]
    best_span = ""

    for cand in candidates:
        cand_n = _norm(cand)
        if not cand_n:
            continue
        # partial_ratio finds the best matching substring in hyp
        s = fuzz.partial_ratio(cand_n, hyp_n)
        if s > best_score:
            best_score = s
            best_alias = cand
            best_span = _best_span(cand_n, hyp_n)

    hit = best_score >= HIT_THRESHOLD
    partial = (not hit) and (best_score >= PARTIAL_THRESHOLD)
    return LocalityScore(best_score, best_alias, best_span, hit, partial)


def _best_span(needle: str, haystack: str) -> str:
    """Return the substring of haystack that best aligns to needle (cheap heuristic)."""
    nw = len(needle)
    if nw >= len(haystack):
        return haystack
    best = (0, haystack[:nw])
    for i in range(0, len(haystack) - nw + 1):
        window = haystack[i:i + nw]
        s = fuzz.ratio(needle, window)
        if s > best[0]:
            best = (s, window)
    return best[1]
