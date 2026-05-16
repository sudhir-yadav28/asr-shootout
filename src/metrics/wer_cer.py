"""WER / CER against ground-truth reference. Picks the best of {Roman, Devanagari} refs.

Why best-of: Hinglish has no canonical spelling. Models like Whisper output Devanagari,
Deepgram outputs Roman, Sarvam may output either. Penalising a correct transcription
because it chose a different script is misleading — we score against whichever ref matches.
"""
import re
import unicodedata
from jiwer import wer, cer


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s).lower().strip()
    # collapse punctuation/whitespace
    s = re.sub(r"[।,.!?;:\"'()\[\]{}\-–—]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def best_wer_cer(hyp: str, refs: list[str]) -> tuple[float, float, str]:
    """Returns (best_wer, best_cer, best_ref_used)."""
    hyp_n = normalize(hyp)
    if not hyp_n:
        return 1.0, 1.0, refs[0] if refs else ""
    best = (float("inf"), float("inf"), refs[0])
    for ref in refs:
        ref_n = normalize(ref)
        if not ref_n:
            continue
        try:
            w = wer(ref_n, hyp_n)
            c = cer(ref_n, hyp_n)
        except Exception:
            continue
        if w < best[0]:
            best = (w, c, ref)
    return best
