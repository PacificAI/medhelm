---
title: PhysicianBench
---

# PhysicianBench

PhysicianBench is an **EHR-agent** evaluation of physician tasks (data retrieval, orders, documentation) against a FHIR server in Docker. It is **not** OpenAI HealthBench (`health_bench`).

This MedHELM scenario (`physician_bench`) wraps a sibling [`PhysicianBench`](https://github.com/healthrex/PhysicianBench) checkout. **One MedHELM instance is one task** (one Docker FHIR container + MiniAgent episode + pytest). v1 has **100 tasks** and **670 pytest checkpoints**. Checkpoints are scored *inside* each task; they are not separate MedHELM instances.

| You set | What runs |
| --- | --- |
| `--max-eval-instances 1` | 1 task (sampled from the loaded pool) |
| `--max-eval-instances 10` | 10 tasks |
| `--max-eval-instances 100` and no `task_ids` | all 100 v1 tasks |

The outer HELM request is routed to `pb/harness`. MiniAgent uses the **evaluated** MedHELM model (for example `openai/gpt-5-mini`). Leaderboard rows use that model name, not `pb/harness`.

## Prerequisites

- Docker Desktop (or another Docker daemon) **running**
- Python 3.10+ via the MedHELM `.venv` (do not use a system Python 3.7)
- [`uv`](https://docs.astral.sh/uv/)
- A sibling PhysicianBench checkout next to `medhelm/`
- The FHIR image archive `physicianbench-fhir-v1.tar.gz` from [Stanford Redivis](https://stanford.redivis.com/datasets/a0ek-0ad8tjsw9)
- An LLM API key (OpenRouter, Anthropic, or OpenAI) for real episodes
- Node.js 20+ **only** if you will open `helm-server` from a git clone (the React UI is not in git)

## Install

From a workspace that contains both checkouts:

```sh
export PHYSICIAN_BENCH_ROOT=/path/to/PhysicianBench

# PhysicianBench harness
cd "$PHYSICIAN_BENCH_ROOT"
uv sync
gunzip -c physicianbench-fhir-v1.tar.gz | docker load   # once; image tag fhir-full:v1

# MedHELM wrap
cd /path/to/medhelm
uv pip install -e ".[physician-bench]"
uv pip install -e "$PHYSICIAN_BENCH_ROOT"
```

After `uv pip install`, CLIs live in `medhelm/.venv/bin/`. Either activate that venv or call the binaries by path. Bare `medhelm-run` fails with `command not found` if the venv is not on `PATH`. Prefer `.venv/bin/medhelm-run` over `uv run medhelm-run` on Intel macOS if lockfile resolution fails on Torch wheels.

### Web UI (git clone only)

`helm-server` returns **404 on `/`** until the frontend is built:

```sh
cd helm-frontend
npm install
npm run build -- --outDir '../src/helm/benchmark/static_build' --emptyOutDir
cd ..
```

## Configure

Create `PhysicianBench/.env` with **one** backend (priority: OpenRouter > Anthropic > OpenAI):

```sh
OPENROUTER_API_KEY=sk-or-...
# or ANTHROPIC_API_KEY=...
# or OPENAI_API_KEY=...
```

MiniAgent **and** pytest LLM-judge checkpoints use these keys. Real episodes need keys for evaluation, not only for the agent.

Point MedHELM at the harness:

```sh
export PHYSICIAN_BENCH_ROOT=/path/to/PhysicianBench
```

You can instead pass `pb_root=` on the run entry. If neither is set, the scenario looks for `../PhysicianBench` relative to the `medhelm/` working directory.

### Models

`--model` / `--model_deployment` must be **registered MedHELM ids**. MiniAgent then gets a PhysicianBench `--model` string from `src/helm/benchmark/static/physician_bench_model_map.yaml` (exact `model` / `model_deployment`, else longest `model_prefix`). Unknown models **fail closed**.

| What you want | MedHELM `model=` and `model_deployment=` | MiniAgent id |
| --- | --- | --- |
| Pipeline check (no Docker, no API) | `simple/model1` | stub (synthetic zeros) |
| GPT-5 mini via OpenRouter | `openai/gpt-5-mini` | `openai/gpt-5-mini` |
| Same model, dated HELM snapshot | `openai/gpt-5-mini-2025-08-07` | `openai/gpt-5-mini` |
| Other OpenAI-prefix HELM models | e.g. `openai/gpt-4o-2024-05-13` | the evaluated id as-is |

`openai/gpt-5-mini` (OpenRouter slug) is registered as a MedHELM alias so the same id works in native PhysicianBench and in `medhelm-run`.

## Run

Always from `medhelm/`, with Docker running for real episodes, and `--num-threads 1`. Each task binds host port **18080**. Episodes are serial even if you pass a higher `--num-threads` (you will see a warning). `--num-threads 1` avoids that warning.

Commas separate run-entry keys. Multiple task slugs use `+`.

### One task

```sh
cd /path/to/medhelm
export PHYSICIAN_BENCH_ROOT=/path/to/PhysicianBench

.venv/bin/medhelm-run --run-entries \
  "physician_bench:task_ids=aortic_aneurysm_cad,max_steps=30,reasoning_effort=medium,model=openai/gpt-5-mini,model_deployment=openai/gpt-5-mini" \
  --suite pb-poc --max-eval-instances 1 --num-threads 1
```

You should see `1 instances, 0 train instances, 1/1 eval instances`, then Docker start, MiniAgent, and pytest for that task’s checkpoints (for example 6 tests on `aortic_aneurysm_cad`).

### Several named tasks

```sh
.venv/bin/medhelm-run --run-entries \
  "physician_bench:task_ids=aortic_aneurysm_cad+chronic_cough_geriatric+postmenopausal_bleeding,max_steps=30,reasoning_effort=medium,model=openai/gpt-5-mini,model_deployment=openai/gpt-5-mini" \
  --suite pb-poc --max-eval-instances 3 --num-threads 1
```

Set `--max-eval-instances` to at least the number of named tasks. If it is smaller, HELM **samples** from the named pool (seed 0).

### Ten tasks (or any N) from the full set

Omit `task_ids` so all **100** v1 tasks are loaded, then cap with `--max-eval-instances`:

```sh
.venv/bin/medhelm-run --run-entries \
  "physician_bench:max_steps=30,reasoning_effort=medium,model=openai/gpt-5-mini,model_deployment=openai/gpt-5-mini" \
  --suite pb-poc --max-eval-instances 10 --num-threads 1
```

Log line: `100 instances, 0 train instances, 10/100 eval instances`.

### All 100 tasks

```sh
.venv/bin/medhelm-run --run-entries \
  "physician_bench:reasoning_effort=medium,model=openai/gpt-5-mini,model_deployment=openai/gpt-5-mini" \
  --suite pb-poc --max-eval-instances 100 --num-threads 1
```

This is slow: one Docker container per task, torn down before the next. Default `max_steps` is 100 if you omit it.

### Pipeline check (no Docker, no API)

```sh
.venv/bin/medhelm-run --run-entries \
  "physician_bench:task_ids=aortic_aneurysm_cad,model=simple/model1,model_deployment=simple/model1" \
  --suite pb-poc --max-eval-instances 1 --num-threads 1
```

`simple/model1` is a **stub**. It does not start FHIR and writes a zero score. Use it only to verify scenario wiring.

### Native PhysicianBench (optional)

To run the harness without MedHELM:

```sh
cd "$PHYSICIAN_BENCH_ROOT"
uv run python scripts/run_task.py tasks/v1/aortic_aneurysm_cad \
  --model openai/gpt-5-mini --reasoning-effort medium --max-steps 30

# two or three tasks
bash scripts/run_batch_task.sh \
  --model openai/gpt-5-mini --reasoning-effort medium --max-steps 30 \
  aortic_aneurysm_cad chronic_cough_geriatric
```

## Summarize and view

After `medhelm-run`:

```sh
cd /path/to/medhelm
.venv/bin/helm-summarize --suite pb-poc -o ./benchmark_output
.venv/bin/helm-server --suite pb-poc -o ./benchmark_output --port 8000
```

Open [http://localhost:8000/](http://localhost:8000/) (not `0.0.0.0`). PhysicianBench is under **Clinical Decision Support**. Scores are attributed to the evaluated model (for example `openai/gpt-5-mini`).

If `/` is 404, build the frontend (see [Install](#web-ui-git-clone-only)) and restart `helm-server`.

## Scoring

Each task runs **all** of its pytest checkpoints, then MedHELM records:

| Metric | Meaning |
| --- | --- |
| `physician_bench_score` (**PB Score**, leaderboard Accuracy) | Checkpoints passed / checkpoints total on that task (0–1). Across tasks this is the **mean** of those fractions. |
| `physician_bench_pass` (**PB Pass**) | 1 only if **every** checkpoint on the task passed, else 0. Same idea as native `RESULT: FAILED`. |
| `physician_bench_checkpoints_passed` / `_total` | Raw pytest counts |
| `physician_bench_steps` | MiniAgent LLM steps (`trajectory.log` `llm_response` events) |
| `physician_bench_prompt_tokens` / `_completion_tokens` | Sum of MiniAgent prompt / completion tokens across those steps |
| `physician_bench_tokens` | Prompt + completion tokens for the task (one MedHELM instance) |

These token totals are MiniAgent usage only (not pytest LLM-judge calls). HELM suite tables show the **mean** across tasks; `per_instance_stats.json` has the per-task total. Native jobs also write the same fields to `metadata.json`. Do not use HELM’s generic `num_prompt_tokens` / `num_output_tokens` for this scenario: those count the outer `pb/harness` envelope, not the agent.

Example: `trd_refill_review` with 3 passed and 4 failed of 7 checkpoints has PB Score **3/7 ≈ 0.43** and PB Pass **0**.

## Run-spec arguments

Commas separate keys. Multiple task ids use `+`.

| Argument | Default | Role |
| --- | --- | --- |
| `task_ids` | all v1 tasks | Filter, e.g. `aortic_aneurysm_cad` or `aortic_aneurysm_cad+chronic_cough_geriatric` |
| `model` / `model_deployment` | required | Evaluated MedHELM model (see [Models](#models)) |
| `max_steps` | 100 | MiniAgent step cap |
| `reasoning_effort` | unset | `low`, `medium`, or `high` |
| `temperature` | API default | Sampling temperature |
| `pb_root` | `PHYSICIAN_BENCH_ROOT` or sibling checkout | Path to PhysicianBench |
| `version` | `v1` | Task tree under `tasks/` |
| `fhir_image` | `fhir-full:v1` | Docker image |
| `port` | `18080` | Host port for FHIR |

CLI flags:

| Flag | Role |
| --- | --- |
| `--max-eval-instances N` | Run **N tasks** (sampled from the loaded pool) |
| `--num-threads 1` | One episode at a time (required for the shared FHIR port) |
| `--suite` | Output subdirectory under `benchmark_output/runs/` |

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `zsh: command not found: medhelm-run` | Venv not on `PATH`. Use `.venv/bin/medhelm-run` from `medhelm/`. |
| `Model deployment openai/gpt-5-mini not found` | Use a registered id (`openai/gpt-5-mini` after this wrap, or `openai/gpt-5-mini-2025-08-07`). |
| Suite finishes with zeros and no Docker | You used `simple/model1` (stub). Use an `openai/` (or other mapped) model for a real episode. |
| Fatal error from `docker info` | Start Docker Desktop; load `fhir-full:v1`. |
| `1 instances` when you wanted many | `task_ids=` listed one task. Omit `task_ids` for the 100-task pool, then set `--max-eval-instances`. |
| `GET /` 404 from `helm-server` | Build `helm-frontend` into `src/helm/benchmark/static_build`. |
| Port 18080 already allocated | A previous FHIR container is still running. `docker ps` / `docker rm -f` the `fhir-bench-*` container. |
