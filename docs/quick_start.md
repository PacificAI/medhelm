---
layout: default
title: Quick Start
---

# Quick Start

MedHELM is a public Python library with fewer dependencies and straightforward installation.

If you **cloned this repository**, install it in editable mode (recommended for development/contributing). If you want the **released package**, install from PyPI.

## Standard (recommended to start)

Scenarios: **PubMedQA**, **MedCalc-Bench**, **MedicationQA**, **MedHallu**.

```sh
# From this repo (recommended if you cloned the repo)
uv pip install -e .

# Or: from PyPI
# uv pip install medhelm
```

**Quick test** (small model, 2 instances — runs in seconds):

```sh
uv run medhelm-run \
  --run-entries "pubmed_qa:model=openai/gpt2,model_deployment=huggingface/gpt2" \
  --suite my_med_test \
  --max-eval-instances 2
uv run helm-summarize --suite my_med_test -o ./benchmark_output
uv run helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

**Full example** (better quality, 10 instances):

```sh
uv run medhelm-run \
  --run-entries "pubmed_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" \
  --suite my_med_test \
  --max-eval-instances 10
uv run helm-summarize --suite my_med_test -o ./benchmark_output
uv run helm-server --suite my_med_test -o ./benchmark_output --port 8000
```

Then open <http://localhost:8000/> in your browser.

## Clinical NLP tier (summarization)

Adds heavier libraries (bert-score, rouge-score, nltk). **Install may take 2–3 minutes.**

Scenarios: **DischargeMe** (hospital course summaries; requires PhysioNet `data_path`), **ACI-Bench** (clinical transcripts), **Patient-Edu** (simplifying medical jargon).

```sh
uv pip install "medhelm[summarization]"
```

Example (ACI-Bench; runs without extra data):

```sh
uv run medhelm-run \
  --run-entries "aci_bench:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" \
  --suite med_summaries \
  --max-eval-instances 5
uv run helm-summarize --suite med_summaries
uv run helm-server --suite med_summaries
```

## Gated / licensing tier (Drive scenarios)

Adds **gdown** so the code can download data from Google Drive. Install can also take longer.

Scenarios: **MedQA** (USMLE/Board exams), **MedMCQA** (AIIMS/NEET exams).

```sh
uv pip install "medhelm[gated]"
```

Example:

```sh
uv run medhelm-run \
  --run-entries "med_qa:model=qwen/qwen2.5-7b-instruct,model_deployment=huggingface/qwen2.5-7b-instruct" \
  --suite board_exams \
  --max-eval-instances 10
uv run helm-summarize --suite board_exams
uv run helm-server --suite board_exams
```

## PhysicianBench (EHR agent)

Requires a sibling PhysicianBench checkout, Docker, and the FHIR image. See [PhysicianBench](physician_bench.md) for install, API keys, and how to run **one task** vs **all 100**.

From `medhelm/` after `uv pip install -e ".[physician-bench]"`:

```sh
export PHYSICIAN_BENCH_ROOT=/path/to/PhysicianBench

# one named task
.venv/bin/medhelm-run --run-entries \
  "physician_bench:task_ids=aortic_aneurysm_cad,max_steps=30,reasoning_effort=medium,model=openai/gpt-5-mini,model_deployment=openai/gpt-5-mini" \
  --suite pb-poc --max-eval-instances 1 --num-threads 1

# all 100 v1 tasks (omit task_ids)
.venv/bin/medhelm-run --run-entries \
  "physician_bench:reasoning_effort=medium,model=openai/gpt-5-mini,model_deployment=openai/gpt-5-mini" \
  --suite pb-poc --max-eval-instances 100 --num-threads 1
```

## Summary

| Tier | Install | Scenarios |
|------|--------|-----------|
| **Standard** | `uv pip install -e .` (repo) or `uv pip install medhelm` (PyPI) | PubMedQA, MedCalc-Bench, MedicationQA, MedHallu |
| **Summarization** | `uv pip install "medhelm[summarization]"` | DischargeMe (needs data_path), ACI-Bench, Patient-Edu (2–3 min install) |
| **Gated** | `uv pip install "medhelm[gated]"` | MedQA, MedMCQA (Drive) |
| **PhysicianBench** | `uv pip install -e ".[physician-bench]"` plus sibling checkout | EHR-agent tasks; see [PhysicianBench](physician_bench.md) |

You can use `pip install medhelm` (and `pip install "medhelm[summarization]"` / `pip install "medhelm[gated]"`) instead of `uv pip install`; then run with `medhelm-run` (or `helm-run`), `helm-summarize`, and `helm-server`.
