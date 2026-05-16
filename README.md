# ASR Shootout — Bangalore Locality Benchmark

Benchmarks 5 automatic speech recognition systems on natural Hinglish speech containing Bangalore locality names. Dataset is 20 phone-mic recordings under varied conditions (quiet, traffic noise, phone call, whispered, rushed) covering 20 localities — 10 well-known, 10 harder Kannada-origin.

## Systems

| System | Type | Runs on |
|---|---|---|
| Deepgram Nova-3 (multi) | API · generic | local |
| Sarvam Saarika v2.5 | API · India-tuned | local |
| Groq Whisper large-v3 | API · hosted Whisper | local |
| OpenAI Whisper large-v3 | Open-source · generic | Colab T4 |
| AI4Bharat IndicConformer 600M | Open-source · India-tuned | Colab T4 |

## Metrics

- **WER / CER** — best of Roman or Devanagari reference
- **Locality Hit Rate (LHR)** — fuzzy + alias-aware entity capture; the headline metric
- **Phonetic similarity** — Metaphone-based, partial credit for "heard right, spelled wrong"
- **Latency** — wall-clock per request

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys

# put .wav / .m4a recordings into data/audio/
python -m src.pipeline   # runs the 3 API systems

# open notebooks/asr_colab.ipynb on Colab (T4 GPU) for Whisper + IndicConformer
# unzip the resulting transcripts into results/transcripts/

python -m src.analyze    # writes results/metrics_summary.csv + plots
```

## Layout

```
src/runners/    one runner per system
src/metrics/    WER/CER, locality match, phonetic
src/pipeline.py one-command orchestrator (API systems)
src/analyze.py  metrics, plots, failure cases
notebooks/      Colab notebook for GPU models
data/           audio + ground_truth.json
results/        cached transcripts, CSVs, plots
```
