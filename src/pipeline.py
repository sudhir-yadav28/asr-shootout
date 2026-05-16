"""One-command pipeline for the API-based ASR systems (Deepgram, Groq, Sarvam).

Whisper-large-v3 and IndicConformer run on Colab — see notebooks/asr_colab.ipynb.
Their transcripts get dropped into results/transcripts/<model>/ and are picked up
automatically by src/analyze.py.

Usage:
    python -m src.pipeline                 # run all API models on all clips
    python -m src.pipeline --only deepgram # one model
    python -m src.pipeline --force         # ignore cache, re-run
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from src.config import AUDIO_DIR, GROUND_TRUTH_PATH, TRANSCRIPTS_DIR
from src.utils.audio import prep_all
from src.runners.deepgram_runner import DeepgramRunner
from src.runners.groq_runner import GroqWhisperRunner
from src.runners.sarvam_runner import SarvamRunner


API_RUNNERS = {
    "deepgram": DeepgramRunner,
    "groq": GroqWhisperRunner,
    "sarvam": SarvamRunner,
}


def load_ground_truth() -> list[dict]:
    if not GROUND_TRUTH_PATH.exists():
        sys.exit(f"Ground truth file missing: {GROUND_TRUTH_PATH}")
    with GROUND_TRUTH_PATH.open() as f:
        return json.load(f)


def resolve_audio(clip: dict, wavs: list[Path]) -> Path | None:
    """Find the wav file for this clip. Match by id prefix OR by full filename."""
    expected = clip["filename"]
    stem = Path(expected).stem.lower()
    by_name = {w.name.lower(): w for w in wavs}
    if expected.lower() in by_name:
        return by_name[expected.lower()]
    by_stem = {w.stem.lower(): w for w in wavs}
    if stem in by_stem:
        return by_stem[stem]
    # last resort: match by leading numeric id
    prefix = clip["id"] + "_"
    for w in wavs:
        if w.stem.lower().startswith(prefix):
            return w
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(API_RUNNERS.keys()), default=None,
                    help="Run a single API runner instead of all.")
    ap.add_argument("--force", action="store_true",
                    help="Ignore cached transcripts and re-run.")
    args = ap.parse_args()

    clips = load_ground_truth()
    print(f"[pipeline] Loaded {len(clips)} clips from ground_truth.json")

    print(f"[pipeline] Prepping audio in {AUDIO_DIR} ...")
    wavs = prep_all(AUDIO_DIR)
    if not wavs:
        sys.exit(
            f"No audio files found in {AUDIO_DIR}.\n"
            "Drop your 20 recordings there (wav/m4a/mp3) and re-run."
        )
    print(f"[pipeline] Found {len(wavs)} audio files.")

    runners_to_run = (
        {args.only: API_RUNNERS[args.only]} if args.only else API_RUNNERS
    )

    for runner_key, runner_cls in runners_to_run.items():
        print(f"\n[{runner_key}] Initialising ...")
        try:
            runner = runner_cls(TRANSCRIPTS_DIR)
        except Exception as e:
            print(f"[{runner_key}] init failed: {e}. Skipping.")
            continue
        ok, err = 0, 0
        for clip in tqdm(clips, desc=f"{runner_key}"):
            audio_path = resolve_audio(clip, wavs)
            if audio_path is None:
                tqdm.write(f"  [{clip['id']}] no audio file — skipped")
                err += 1
                continue
            res = runner.transcribe(
                clip_id=clip["id"],
                filename=audio_path.name,
                audio_path=audio_path,
                force=args.force,
            )
            if res.error:
                err += 1
                tqdm.write(f"  [{clip['id']}] ERROR: {res.error.splitlines()[0]}")
            else:
                ok += 1
        print(f"[{runner_key}] done. ok={ok} err={err}")

    print("\n[pipeline] All API runs complete.")
    print(f"  Transcripts cached in: {TRANSCRIPTS_DIR}")
    print("  Next: run Colab notebook for Whisper + IndicConformer, then `python -m src.analyze`")


if __name__ == "__main__":
    main()
