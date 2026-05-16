"""Project paths and environment loading."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"

RESULTS_DIR = ROOT / "results"
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"
PER_CLIP_CSV = RESULTS_DIR / "per_clip_metrics.csv"

REPORT_DIR = ROOT / "report"

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

for p in (AUDIO_DIR, TRANSCRIPTS_DIR, PLOTS_DIR, REPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)
