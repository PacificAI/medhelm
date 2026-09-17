---
title: HealthAdminBench
---
# HealthAdminBench

HealthAdminBench is a **computer-use** evaluation of healthcare administration workflows (prior authorization, appeals/denials, DME). The MedHELM scenario name is `health_admin_bench`. It is **not** OpenAI HealthBench (`health_bench`).

Paper: [HealthAdminBench (Bedi et al., 2026)](https://arxiv.org/abs/2604.09937). Upstream harness: [som-shahlab/health-admin-bench](https://github.com/som-shahlab/health-admin-bench).

This page covers how to **clone MedHELM and HealthAdminBench**, **set the HAB path**, **install dependencies**, and **run** the scenario with `medhelm-run`. For general MedHELM install, see [Installation](/installation) and [Quick Start](/quick_start). API keys: [Credentials](/credentials).

## What MedHELM runs

One MedHELM instance is one Playwright episode. The adapter routes the outer request to the internal deployment `hab/harness`. OpenAI-chat-compatible `model_deployment`s run as `HelmBackedAgent` (each browser step is an inner `AutoClient` request). Native HAB agents (RandomAgent, Claude, Gemini) stay an override in `health_admin_bench_model_map.yaml`. The model completion is HAB `EvaluationResult` JSON. The leaderboard metric is `health_admin_bench_score` (subeval points / max, 0–1), under **Administration and Workflow**.

## Get the code

You need **two** checkouts next to each other: MedHELM and HealthAdminBench. Clone HAB from [som-shahlab/health-admin-bench](https://github.com/som-shahlab/health-admin-bench) (`main`). MedHELM calls HAB in-process (`run_task(agent=..., llm_complete=...)`).

```bash
# Parent directory for both repos (adjust as you like)
mkdir -p ~/src && cd ~/src

git clone https://github.com/PacificAI/medhelm.git
git clone https://github.com/som-shahlab/health-admin-bench.git
```

Expected layout:

```text
~/src/
  medhelm/              # PacificAI/medhelm
  health-admin-bench/   # som-shahlab/health-admin-bench
```

The directories do not have to be siblings, but a single parent folder is easiest. Record the **absolute** path to the HAB checkout; MedHELM will not find task JSON files without it.

## Set the HAB path

Export this in every shell that runs `medhelm-run` (or put it in `~/.zshrc` / `~/.bashrc`):

```bash
export HEALTH_ADMIN_BENCH_ROOT="$HOME/src/health-admin-bench"
```

Replace `$HOME/src/health-admin-bench` with the absolute path from `git clone`. You can also pass `hab_root=/absolute/path/to/health-admin-bench` on the run entry instead of the env var.

Check:

```bash
test -f "$HEALTH_ADMIN_BENCH_ROOT/run.py" && echo "HAB root OK"
```

## Install dependencies

MedHELM needs Python **3.12**. HealthAdminBench needs **Python ≥ 3.10**, [uv](https://docs.astral.sh/uv/), and Playwright Chromium. Node.js is only required if you run the local portals or build the `helm-server` UI.

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

### 1. HealthAdminBench harness

```bash
cd "$HEALTH_ADMIN_BENCH_ROOT"
uv sync
uv run hab install          # Playwright Chromium + copy .env.local → .env
```

`hab install` is required even when you only drive HAB from MedHELM: it installs Chromium and creates HAB `.env`.

### 2. MedHELM with the HealthAdminBench extra

HealthAdminBench is **not** in the default MedHELM install (`uv pip install -e .`). The scenario code lives in this repo, but Playwright and HAB itself are opt-in — the same idea as `[summarization]` and `[gated]` in [Installation](/installation).

Install MedHELM **and** the HAB package into the **same** MedHELM virtualenv. Playwright must be available in that env, not only in HAB’s `.venv`.

```bash
cd /path/to/medhelm
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[health-admin-bench]"   # MedHELM + Playwright extras (not default)
uv pip install -e "$HEALTH_ADMIN_BENCH_ROOT"  # HAB harness (separate repo)
python -m playwright install chromium
```

| Extra | What it adds |
| --- | --- |
| `[health-admin-bench]` | Playwright, jmespath, loguru, pydantic-settings |
| Editable HAB install | `harness` / `run.py` imports used by `HealthAdminBenchClient` |

### 3. Leaderboard UI (optional, before `helm-server`)

A git clone does not ship a pre-built frontend. If you want the web UI:

```bash
cd /path/to/medhelm/helm-frontend
npm install
npm run build -- --outDir '../src/helm/benchmark/static_build' --emptyOutDir
```

## Credentials

MedHELM agent and judge HTTP use **`prod_env/credentials.conf`** (HOCON), not HAB `.env` and not `OPENAI_API_KEY`. Create `prod_env/` next to `medhelm-run` if needed (the directory is gitignored).

| Key | Used by |
| --- | --- |
| `openaiApiKey` | Public OpenAI deployments (`openai/gpt-4o-…`) |
| `xaiApiKey` | Grok (`xai/grok-…`) |
| `stanfordhealthcareApiKey` + `stanfordhealthcareEndpoint` | Package judges YAML and `stanfordhealthcare/*` Azure/APIM |
| `azureApiKey` + `azureEndpoint` | A local `azure/…` row in `prod_env/model_deployments.yaml` |

Example (public OpenAI):

```text
openaiApiKey: sk-...
xaiApiKey: xai-...
```

Example (your Azure OpenAI resource; prefix of the deployment name is the credential prefix):

```text
azureApiKey: ...
azureEndpoint: https://<resource>.openai.azure.com
```

You do **not** need `openaiApiKey` if the **agent and judge** both use Azure (`azure/…` or `stanfordhealthcare/…`).

HAB `.env` is only for **`hab_native`** agents (Claude / Gemini) and the OpenRouter judge **fallback** when the envelope has no MedHELM judge:

```bash
cd "$HEALTH_ADMIN_BENCH_ROOT"
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
echo 'GEMINI_API_KEY=...'           >> .env
echo 'OPENROUTER_API_KEY=sk-or-...' >> .env   # only if you skip jury_config_path
```

Do not commit `.env` or `credentials.conf`.

## Portals

Hosted portals default to `https://emrportal.vercel.app`. To run a local EMR portal:

```bash
cd "$HEALTH_ADMIN_BENCH_ROOT/benchmark/v2/portals"
npm install && npm run dev
```

Then pass `env_base_url=http://localhost:3002` in the run entry.

## Parallelism

HealthAdminBench episodes are **always serial**. MedHELM `--num-threads` (default 4) is an in-process thread pool over `make_request()`, not HAB `--max-parallel` / `-j`. Playwright’s sync API and the harness `os.chdir` cannot overlap. If you pass `--num-threads 4`, MedHELM ignores it for HAB runs and still executes one episode at a time (you will see a warning). `--num-threads 1` avoids the warning.

## Smoke test (no frontier API)

`simple/model1` maps to HealthAdminBench `RandomAgent`. Expect a failed episode on `emr-easy-1`: the random agent does not complete the workflow. This only checks that MedHELM can find HAB, launch Chromium, and write stats.

The **default judge is still Stanford Healthcare**. Without `jury_config_path` (or SHC keys), LLM-judge subevals score 0 with `[COMPLETE_ERROR]` and the episode still finishes. Pass a public or Azure judges YAML if you want a real judge on this smoke.

```bash
export HEALTH_ADMIN_BENCH_ROOT=/absolute/path/to/health-admin-bench
cd /path/to/medhelm
source .venv/bin/activate

medhelm-run --run-entries \
  "health_admin_bench:task_ids=emr-easy-1,max_steps=3,model=simple/model1,model_deployment=simple/model1" \
  --suite hab-poc --max-eval-instances 1 --num-threads 1
helm-summarize --suite hab-poc -o ./benchmark_output
helm-server --suite hab-poc -o ./benchmark_output --port 8000
```

Open http://localhost:8000. HealthAdminBench is under **Administration and Workflow**.

Optional standalone check (HAB CLI, same task):

```bash
cd "$HEALTH_ADMIN_BENCH_ROOT"
uv run hab run --model random --task emr-easy-1 --observation-mode axtree_only --prompt-mode general
```

## Evaluated model (HelmBackedAgent)

If the run entry’s `model_deployment` exists and its client is `OpenAIClient` or a chat-compatible subclass (`GrokChatClient`, `AzureOpenAIClient`, OpenAI-API `base_url`), HealthAdminBenchClient builds `HelmBackedAgent`. No map row is required.

Public OpenAI (needs `openaiApiKey`):

```bash
export HEALTH_ADMIN_BENCH_ROOT=/absolute/path/to/health-admin-bench
cd /path/to/medhelm
source .venv/bin/activate

medhelm-run --run-entries \
  "health_admin_bench:task_ids=emr-easy-1,observation_mode=axtree_only,jury_config_path=src/helm/benchmark/static/health_admin_bench_judges.yaml,model=openai/gpt-4o-2024-05-13,model_deployment=openai/gpt-4o-2024-05-13" \
  --suite hab-poc --max-eval-instances 1 --num-threads 1
```

Grok:

```bash
medhelm-run --run-entries \
  "health_admin_bench:task_ids=emr-easy-1,observation_mode=axtree_only,jury_config_path=src/helm/benchmark/static/health_admin_bench_judges.yaml,model=xai/grok-4-0709,model_deployment=xai/grok-4-0709" \
  --suite hab-poc --max-eval-instances 1 --num-threads 1
```

Local OpenAI-compatible server: add a row in `prod_env/model_deployments.yaml` with `helm.clients.openai_client.OpenAIClient` and `base_url`. See [Adding New Models](/adding_new_models).

Local **Azure OpenAI** (no public `openaiApiKey`): add a deployment whose name prefix matches the credential keys (`azure` → `azureApiKey` / `azureEndpoint`):

```yaml
# prod_env/model_deployments.yaml
model_deployments:
  - name: azure/gpt-5-mini
    model_name: openai/gpt-5-mini-2025-08-07
    tokenizer_name: openai/o200k_base
    max_sequence_length: 400000
    client_spec:
      class_name: "helm.clients.azure_openai_client.AzureOpenAIClient"
      args:
        azure_openai_deployment_name: gpt-5-mini   # Azure deployment name
        api_version: "2024-06-01"
```

Point the judge at the same deployment (example `prod_env/health_admin_bench_judges.yaml`):

```yaml
judges:
  - name: gpt
    model: openai/gpt-5-mini-2025-08-07
    model_deployment: azure/gpt-5-mini
```

```bash
medhelm-run --run-entries \
  "health_admin_bench:task_ids=emr-easy-1,observation_mode=axtree_only,is_gui=true,jury_config_path=prod_env/health_admin_bench_judges.yaml,model=openai/gpt-5-mini-2025-08-07,model_deployment=azure/gpt-5-mini" \
  --suite hab-azure --max-eval-instances 1 --num-threads 1
```

`is_gui=true` is a **run-entry** argument (`headless=False`). There is no `medhelm-run --is-gui` flag.

Stanford Healthcare Azure GPT-5 (needs `stanfordhealthcareApiKey` + `stanfordhealthcareEndpoint`): `model_deployment=stanfordhealthcare/gpt-5-2025-08-07`. Omit `jury_config_path` to reuse the package SHC judges.

Non-OpenAI-chat clients fail closed unless `src/helm/benchmark/static/health_admin_bench_model_map.yaml` sets `backend: hab_native` (RandomAgent, Claude, Gemini).

## Judges

Reuse HealthBench `get_annotator_models_from_config(jury_config_path)`. The first YAML judge is copied onto the episode envelope. HAB `LLMJudge` still runs **inside** `evaluate_episode` (portal `{{jmespath}}` state, JSON parse, majority vote). Only the HTTP call goes through `AutoClient` via `llm_complete`.

HAB may pass `system`, `temperature`, `max_tokens`, and a native judge `model` id (often `gpt-5.4` from the task JSON). MedHELM uses **envelope** `judge_model` / `judge_model_deployment` for the AutoClient `Request` — it does not treat HAB’s native id as a MedHELM model name. A one-argument `complete(prompt) -> str` still works.

If the callback raises (timeout, missing key, Azure/auth), HAB records `[COMPLETE_ERROR] Type: message`, scores that run 0, and continues majority vote. The episode returns a structured eval row instead of aborting `grade()`. Extra HTTP retries stay on HAB’s native path only.

- Default: package `helm.benchmark.scenarios.medhelm.judges.yaml` (Stanford Healthcare → `stanfordhealthcareApiKey`).
- Public OpenAI judge: `jury_config_path=src/helm/benchmark/static/health_admin_bench_judges.yaml` (`openaiApiKey`).
- Local Azure judge: `jury_config_path=prod_env/health_admin_bench_judges.yaml` as above.
- Override: `judge_model` / `judge_model_deployment`.

Do not set the judge to `hab/harness`. Do not use the evaluated model as judge unless the run spec sets them equal on purpose. Keep HAB’s JSON grader contract `{score, reasoning, evidence_quote}` — HealthBench `<score>` tags parse as 0.

## Run-spec arguments

Commas separate args, so multiple task ids use `+`:

```text
health_admin_bench:task_ids=emr-easy-1+emr-easy-2,task_set=prior_auth,difficulty=easy,model=...,model_deployment=...
```

| Argument | Default | Notes |
| --- | --- | --- |
| `task_ids` | (all matching filters) | e.g. `emr-easy-1` |
| `task_set` | `prior_auth` | `prior_auth` \| `appeals_denials` \| `dme` \| `all` |
| `difficulty` | `easy` | `easy` \| `medium` \| `hard` \| `all` |
| `prompt_mode` | `general` | `zero_shot` \| `general` \| `task_specific` |
| `observation_mode` | `axtree_only` | `axtree_only` for HelmBackedAgent; `screenshot_only` / `both` are later |
| `action_space` | `dom` | `dom` \| `coordinate` |
| `env_base_url` | `https://emrportal.vercel.app` | local: `http://localhost:3002` |
| `hab_root` | `$HEALTH_ADMIN_BENCH_ROOT` | absolute path to the HAB checkout |
| `max_steps` | HAB defaults (easy ≈ 20) | cap episode length for smoke tests |
| `is_gui` | `false` | `true` shows Chromium. Put this **in the run entry**, not as `medhelm-run --is-gui`. |
| `jury_config_path` | package `judges.yaml` (SHC) | see Judges above |
| `judge_model` / `judge_model_deployment` | first judge in that YAML | run-entry override |

`--max-eval-instances N` still applies after the scenario loads tasks. For all 20 prior-auth easy tasks use `task_ids=emr-easy-1+…+emr-easy-20` (or `task_set=prior_auth,difficulty=easy`) and `--max-eval-instances 20`.

After a run:

```bash
helm-summarize --suite <suite> -o ./benchmark_output
helm-server --suite <suite> -o ./benchmark_output --port 8000
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Task JSON / `run.py` not found | Export `HEALTH_ADMIN_BENCH_ROOT` or pass `hab_root=`. MedHELM also looks at `./health-admin-bench` and `../health-admin-bench` relative to cwd (not the package install path). Clone [som-shahlab/health-admin-bench](https://github.com/som-shahlab/health-admin-bench) (`main`). |
| `ModuleNotFoundError: harness` / HAB imports | `uv pip install -e "$HEALTH_ADMIN_BENCH_ROOT"` into the MedHELM venv. |
| Playwright browser missing | `source .venv/bin/activate && python -m playwright install chromium` (MedHELM venv). |
| `The api_key client option must be set` / `openaiApiKey should be specified` | HelmBackedAgent reads `prod_env/credentials.conf`, not HAB `.env`. Match the key to the deployment prefix (`openaiApiKey`, `azureApiKey`, `stanfordhealthcareApiKey`). |
| Judge looks for Stanford credentials / `stanfordhealthcareApiKey` | Default judges YAML is SHC. Pass `jury_config_path=` to a public OpenAI or local Azure judges file. |
| `[COMPLETE_ERROR]` / judge `run_scores=[0,0,0]` | Callback failed (missing key, timeout, auth). Episode still finishes; fix credentials or `jury_config_path`. |
| `prod_env/cache` does not exist | Create it once (`mkdir -p prod_env/cache`). Cache paths are stored absolute so HAB `os.chdir` does not resolve `prod_env/cache` under the HAB checkout. |
| `OPENROUTER_API_KEY is required` | Only if no MedHELM judge is on the envelope. Prefer `jury_config_path`. |
| `no agent for model X` / not OpenAI-chat-compatible | Use `OpenAIClient` / `GrokChatClient` / `AzureOpenAIClient`, or a `hab_native` map row. |
| missing `evaluated_model` / `evaluated_model_deployment` | Set `model=` and `model_deployment=` on the run entry. The client fails with that message instead of a generic no-agent miss. |
| invalid `prompt_mode` / `observation_mode` / `action_space` | Use `zero_shot` \| `general` \| `task_specific`, `screenshot_only` \| `axtree_only` \| `both`, `dom` \| `coordinate`. The client lists those values in the error. |
| `must not be hab/harness` | Agent and judge deployments cannot be `hab/harness`. |
| GUI never opens | `is_gui=true` inside the run-entry string, not `--is-gui` on `medhelm-run`. |
| `ModuleNotFoundError: helm.benchmark.static_build` | Build the frontend (step 3). |
| Warning about `--num-threads` | Harmless; HAB episodes stay serial. Use `--num-threads 1` to silence it. |
| Hosted portal changed | Pin `env_base_url` or run local portals on `:3002`. |

## Summary

| Step | Command |
| --- | --- |
| Clone MedHELM | `git clone https://github.com/PacificAI/medhelm.git` |
| Clone HAB | `git clone https://github.com/som-shahlab/health-admin-bench.git` |
| Point MedHELM at HAB | `export HEALTH_ADMIN_BENCH_ROOT=/absolute/path/to/health-admin-bench` |
| Install HAB | `cd "$HEALTH_ADMIN_BENCH_ROOT" && uv sync && uv run hab install` |
| Install MedHELM extra | `uv pip install -e ".[health-admin-bench]" && uv pip install -e "$HEALTH_ADMIN_BENCH_ROOT"` |
| Smoke test | `medhelm-run --run-entries "health_admin_bench:task_ids=emr-easy-1,max_steps=3,model=simple/model1,model_deployment=simple/model1" --suite hab-poc --max-eval-instances 1 --num-threads 1` |
