"""Score every cached transcript and emit summary CSV + plots + failure cases.

Picks up transcripts from results/transcripts/<model>/<clip_id>.json — so this works
whether the model ran locally (API runners) or on Colab (Whisper / IndicConformer).
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    GROUND_TRUTH_PATH, TRANSCRIPTS_DIR, METRICS_CSV, PER_CLIP_CSV, PLOTS_DIR
)
from src.metrics.wer_cer import best_wer_cer
from src.metrics.locality_match import score_locality
from src.metrics.phonetic import normalised_phonetic_similarity


def _load_ground_truth() -> dict[str, dict]:
    with GROUND_TRUTH_PATH.open() as f:
        clips = json.load(f)
    return {c["id"]: c for c in clips}


def _discover_models() -> list[str]:
    if not TRANSCRIPTS_DIR.exists():
        return []
    return sorted([d.name for d in TRANSCRIPTS_DIR.iterdir() if d.is_dir()])


def _load_transcripts(model: str) -> dict[str, dict]:
    out = {}
    for f in (TRANSCRIPTS_DIR / model).glob("*.json"):
        with f.open() as h:
            data = json.load(h)
        out[data["clip_id"]] = data
    return out


def score_all() -> pd.DataFrame:
    gt = _load_ground_truth()
    models = _discover_models()
    if not models:
        raise SystemExit(
            f"No transcripts in {TRANSCRIPTS_DIR}. Run `python -m src.pipeline` first."
        )

    rows = []
    for model in models:
        transcripts = _load_transcripts(model)
        for clip_id, clip in gt.items():
            t = transcripts.get(clip_id)
            if t is None:
                rows.append(_blank_row(model, clip, reason="no_transcript"))
                continue
            if t.get("error"):
                rows.append(_blank_row(model, clip, reason="error",
                                       hyp="", latency=t.get("latency_s", 0.0),
                                       error=t["error"].splitlines()[0]))
                continue

            hyp = t.get("text", "") or ""
            refs = [clip["transcript_roman"], clip["transcript_devanagari"]]
            wer_v, cer_v, ref_used = best_wer_cer(hyp, refs)
            loc = score_locality(hyp, clip["locality"], clip.get("locality_aliases", []))
            phon = normalised_phonetic_similarity(clip["locality"], loc.matched_text)

            rows.append({
                "model": model,
                "clip_id": clip_id,
                "locality": clip["locality"],
                "condition": clip["condition"],
                "difficulty": clip["difficulty"],
                "hypothesis": hyp,
                "ref_used": "devanagari" if ref_used == clip["transcript_devanagari"] else "roman",
                "wer": round(wer_v, 4),
                "cer": round(cer_v, 4),
                "locality_score": round(loc.score, 1),
                "locality_hit": loc.hit,
                "locality_partial": loc.partial,
                "matched_alias": loc.matched_alias,
                "matched_span": loc.matched_text,
                "phonetic_sim": round(phon, 3),
                "latency_s": round(t.get("latency_s", 0.0), 3),
                "error": None,
            })
    df = pd.DataFrame(rows)
    return df


def _blank_row(model, clip, *, reason: str, hyp: str = "", latency: float = 0.0,
               error: str | None = None) -> dict:
    return {
        "model": model,
        "clip_id": clip["id"],
        "locality": clip["locality"],
        "condition": clip["condition"],
        "difficulty": clip["difficulty"],
        "hypothesis": hyp,
        "ref_used": None,
        "wer": 1.0,
        "cer": 1.0,
        "locality_score": 0.0,
        "locality_hit": False,
        "locality_partial": False,
        "matched_alias": None,
        "matched_span": "",
        "phonetic_sim": 0.0,
        "latency_s": latency,
        "error": error or reason,
    }


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model").agg(
        n=("clip_id", "count"),
        wer=("wer", "mean"),
        cer=("cer", "mean"),
        locality_hit_rate=("locality_hit", "mean"),
        locality_partial_rate=("locality_partial", "mean"),
        phonetic_sim=("phonetic_sim", "mean"),
        latency_mean_s=("latency_s", "mean"),
        latency_p90_s=("latency_s", lambda s: s.quantile(0.9)),
        errors=("error", lambda s: s.notna().sum()),
    ).round(3).sort_values("locality_hit_rate", ascending=False)
    return g


def make_plots(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Headline metric comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["locality_hit_rate", "wer", "cer", "phonetic_sim"]
    summary[metrics].plot(kind="bar", ax=ax)
    ax.set_title("ASR shootout — headline metrics (lower WER/CER better, higher LHR/Phonetic better)")
    ax.set_ylabel("score")
    ax.set_xlabel("")
    ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_headline_metrics.png", dpi=150)
    plt.close()

    # 2. Latency (API models only — local models on Colab have non-comparable wall-clock)
    api_like = summary[summary["latency_mean_s"] > 0.05]
    if not api_like.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        api_like[["latency_mean_s", "latency_p90_s"]].plot(kind="bar", ax=ax)
        ax.set_title("Per-clip latency (seconds) — API-hosted models")
        ax.set_ylabel("seconds")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "02_latency.png", dpi=150)
        plt.close()

    # 3. Locality hit rate by condition (where do models break?)
    pivot = df.pivot_table(
        index="condition", columns="model", values="locality_hit", aggfunc="mean"
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Locality hit rate by recording condition")
    ax.set_ylabel("hit rate")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_hit_by_condition.png", dpi=150)
    plt.close()

    # 4. Locality hit rate by difficulty (easy / medium / hard Kannada names)
    pivot2 = df.pivot_table(
        index="difficulty", columns="model", values="locality_hit", aggfunc="mean"
    ).fillna(0.0).reindex(["easy", "medium", "hard"])
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot2.plot(kind="bar", ax=ax)
    ax.set_title("Locality hit rate by name difficulty")
    ax.set_ylabel("hit rate")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_hit_by_difficulty.png", dpi=150)
    plt.close()


def failure_cases(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Clips where the majority of models missed the locality. Sorted by miss count."""
    miss = df.groupby("clip_id").agg(
        locality=("locality", "first"),
        condition=("condition", "first"),
        difficulty=("difficulty", "first"),
        n_models=("model", "count"),
        n_hits=("locality_hit", "sum"),
        n_partial=("locality_partial", "sum"),
    )
    miss["n_misses"] = miss["n_models"] - miss["n_hits"] - miss["n_partial"]
    miss = miss.sort_values("n_misses", ascending=False).head(k)
    return miss


def main():
    df = score_all()
    PER_CLIP_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PER_CLIP_CSV, index=False)

    summary = summarise(df)
    summary.to_csv(METRICS_CSV)

    make_plots(df, summary)

    fails = failure_cases(df)
    fails.to_csv(PLOTS_DIR.parent / "failure_cases.csv")

    print("\n=== Summary ===")
    print(summary.to_string())
    print(f"\nPer-clip CSV: {PER_CLIP_CSV}")
    print(f"Summary CSV:  {METRICS_CSV}")
    print(f"Plots:        {PLOTS_DIR}")
    print(f"Failure cases:{PLOTS_DIR.parent / 'failure_cases.csv'}")


if __name__ == "__main__":
    main()
