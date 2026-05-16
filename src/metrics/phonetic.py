"""Phonetic distance — gives partial credit when model heard the right sound
but wrote a different spelling (e.g. Hesarghatta vs Hesaraghatta).

Uses Metaphone (English-centric, imperfect for Indic but a useful weak signal).
Computed only on the matched span from locality_match, against the Roman locality name.
"""
import jellyfish


def metaphone_distance(a: str, b: str) -> int:
    """Edit distance between Metaphone codes. 0 = same sounding."""
    if not a or not b:
        return max(len(a or ""), len(b or ""))
    ma = jellyfish.metaphone(a)
    mb = jellyfish.metaphone(b)
    return jellyfish.levenshtein_distance(ma, mb)


def normalised_phonetic_similarity(a: str, b: str) -> float:
    """1.0 = sounds identical, 0.0 = totally different. Roman-script inputs only."""
    if not a or not b:
        return 0.0
    ma = jellyfish.metaphone(a) or a
    mb = jellyfish.metaphone(b) or b
    dist = jellyfish.levenshtein_distance(ma, mb)
    denom = max(len(ma), len(mb), 1)
    return max(0.0, 1.0 - dist / denom)
