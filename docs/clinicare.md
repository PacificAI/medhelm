---
title: CliniCARE-Bench
---
# CliniCARE-Bench

CliniCARE-Bench is an **agentic** clinical benchmark over MIMIC-IV: 750 cases (25 scenarios × 30 patients). In each case an agent investigates one real patient record through a governed tool surface and returns a four-way verdict (Yes / No / Indeterminate – lack of data / Indeterminate – medical ambiguity) with cited evidence. The MedHELM scenario name is `clinicare`.

Upstream: [github.com/scaleapi/clinicare](https://github.com/scaleapi/clinicare). It is a **gated** benchmark: running it needs PhysioNet MIMIC-IV credentials, and the cohort file is distributed by Pacific AI on request.

## What MedHELM runs

One MedHELM instance is one CliniCARE task, and one request is one full agent episode. CliniCARE stays a separate project: MedHELM wraps it, and never imports its code.

```
medhelm-run clinicare:model=<model>
  → CliniCAREScenario       one instance per built task dir (instruction.md is the prompt)
  → CliniCAREAdapter        model stays the evaluated model; request goes to clinicare/harness
  → CliniCAREClient         runs  <CLINICARE_ROOT>/.venv/bin/python scripts/run_case.py ...
                            (harbor + Docker + the MIMIC tool sidecar), then reads the trial
  → CliniCARE metrics       clinicare_score etc., from the trial's verifier/reward.json
```

`scripts/run_case.py` is the contract between the two projects. It runs one task with one CliniCARE system and prints the trial as JSON. Any change inside CliniCARE that keeps that command and its output stable reaches MedHELM without a MedHELM change.

The client can also **replay** a finished CliniCARE job dir instead of running it (`job_dir=` on the run entry). That mode needs no Docker, MIMIC or API keys.

## Get the code

```
~/src/
  medhelm/      # PacificAI/medhelm
  clinicare/    # scaleapi/clinicare (must include scripts/run_case.py)
```

## Set the CliniCARE path

```bash
export CLINICARE_ROOT=/absolute/path/to/clinicare
test -f "$CLINICARE_ROOT/scripts/run_case.py" && echo ok
```

`clinicare_root=` on the run entry overrides it, but keep paths **out** of run entries you publish (see [Privacy and publishing](#privacy-and-publishing)).

## One-time CliniCARE setup

This happens once, in the CliniCARE checkout, following its README and `DATA_ACCESS.md`. MedHELM needs no CliniCARE packages in its own environment.

1. `uv sync`, which creates `$CLINICARE_ROOT/.venv`. MedHELM runs cases with that interpreter.
2. **Cohort bundle** (`full750.csv`, gold labels): request it from Pacific AI, which sends a time-limited download link.
3. **Policy corpus** (67 guideline documents): assemble it with `construction/scripts/acquire_corpus.py`.
4. **MIMIC-IV Parquet** (~8 GB, 41 tables): needs PhysioNet credentialing. Use local disk, not network storage, since every tool call is a Parquet read.
5. Copy `.env.template` to `.env`. Set `MIMIC_DATA_ROOT`, `CLINICARE_COHORT_DIR`, `CLINICARE_CORPUS_DIR`, a non-empty `MIMIC_API_SECRET`, and the model keys from [Credentials](#credentials). `run_case.py` loads `.env` itself.
6. `bash scripts/build_image.sh`, then `uv run python benchmark/scripts/build_tasks.py`. **Set any `*_BASE_URL` before building.** The gateway host is baked into each task's network allowlist at build time.

Rebuild the image and the tasks after pulling a CliniCARE update that changes the image (the trial `Dockerfile` or `packages/mimic_tools`). The image tag is a content hash, so tasks built earlier still point at the old image.

## Credentials

Model credentials live in CliniCARE's `.env`, not in MedHELM's `credentials.conf`. The agent harness (Codex, Claude Code, Gemini CLI, OpenCode) calls the model from inside the trial container.

| system family | key | optional gateway |
| --- | --- | --- |
| `codex-*`, `opencode-*` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` (with `/v1`) |
| `claude-*` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` (no `/v1`) |
| `gemini-*` | `GEMINI_API_KEY` | `GOOGLE_GEMINI_BASE_URL` (`/gemini`) |

LLM-judged scores (`grade=true`) also need `JUDGE_BASE_URL`, `JUDGE_API_KEY` and, optionally, `JUDGE_MODEL`.

## Model map

`src/helm/benchmark/static/clinicare_model_map.yaml` maps each evaluated MedHELM model to the CliniCARE system that runs it. A system is a harness plus a model, defined in CliniCARE's `scripts/run_sweep.py`:

```yaml
models:
  - model: openai/gpt-x         # MedHELM model (the leaderboard row); exact match only
    system: codex-gptx            # CliniCARE system tag
    harbor_model: openai/gpt-x    # what every trial must record as config.agent.model_name
    effort: null
```

The map ships with five models, run by CliniCARE's `--systems medhelm` group:

| MedHELM model | CliniCARE system | harness |
| --- | --- | --- |
| `anthropic/claude-opus-5-5` | `claude-opus55` | Claude Code |
| `google/gemini-3.8-flash` | `gemini-38flash` | Gemini CLI |
| `meta/muse-spark-1.3` | `opencode-musespark13` | OpenCode |
| `openai/gpt-6-astra` | `codex-gpt6astra` | Codex |
| `qwen/qwen3.8-max-0902` | `opencode-qwen38max` | OpenCode |

The two OpenCode systems address their models as routes on an OpenAI-compatible gateway (`fireworks_ai/qwen3p8-max`, `meta_ai/muse-spark-1.3`), so they need `OPENAI_BASE_URL` set to a gateway that serves those names. Add a row for any other model you run. An unmapped model fails closed. Every trial, live or replayed, is checked against `harbor_model`, and a mismatch stops the run rather than publish one model's scores under another's name. The run entry still needs a `model_deployment` registered for the model; it is not called.

## Parallelism

Cases run in parallel: `--num-threads N` means N concurrent episodes, each with its own containers. Start with 4–8 on a laptop and up to 16 on a large machine. Your model provider's rate limit is usually the binding constraint. If you see 429s, lower it rather than relying on retries.

Each case is a harbor job under `$CLINICARE_ROOT/jobs/medhelm/` (override with `jobs_dir=`), named after the system, effort, tasks build and task. Re-running the same suite returns finished cases immediately without re-running them, so an interrupted run resumes for free. A finished case is reused only while its task is byte-identical to the current build. After a CliniCARE update or a task rebuild, `run_case.py` refuses the old trial and the run stops, so point `jobs_dir=` at a fresh directory.

## Smoke test

Replay a provided job dir (no Docker, MIMIC or keys):

```bash
medhelm-run --run-entries "clinicare:tasks_dir=<tasks dir>,job_dir=<job dir>,model=<model>,model_deployment=<model deployment>" \
  --suite clinicare-replay --max-eval-instances 750
```

Replay expects one job dir covering every task in `tasks_dir`, such as a CliniCARE sweep job. Any task without a trial in it counts in `clinicare_trial_failed`. A live MedHELM run writes one job dir per case under `jobs/medhelm/`, so to replay one of those, add `task_ids=` with that case's task id. Replay refuses a job whose trials were built from a different version of their tasks (harbor's task checksum differs), so replay against the tasks build the job ran on.

Run one case live, after the one-time setup. `tasks_dir` can point at a dir holding one task plus `_sidecar.compose.yml`, or pass `task_ids=` with the full build:

```bash
medhelm-run --run-entries "clinicare:task_ids=<task id>,model=<model>,model_deployment=<model deployment>" \
  --suite clinicare-smoke --max-eval-instances 1 --num-threads 1
helm-summarize --suite clinicare-smoke
helm-server --suite clinicare-smoke
```

Check `stats.json`: `clinicare_tool_calls` > 0 and `clinicare_evidence_calls_auth_error` == 0. A run with zero tool calls means the MIMIC sidecar never came up.

## Grading

`clinicare_score` and the other deterministic metrics come from the in-container verifier and need no judge. `grade=true` additionally runs CliniCARE's LLM judges after each case. That fills `clinicare_process_score` and `clinicare_policy_support`. Both are reported separately and never blended into the main score, and both are absent when grading did not run.

If `grade=true` was requested but a judge could not score a case, the case keeps its score, the run continues, and the case counts in `clinicare_grading_failed`. A judge skipping a case by design (no report, a policy-off task, no rubrics of that kind) is not a failure. A case already graded by an earlier run counts as graded. In replay mode MedHELM runs no judges, so a job must be graded in CliniCARE first.

## Warnings

Every CliniCARE warning and per-case error starts with a tag that says which kind of problem it is, then names what to check and the file to open. Each kind is logged once per run, naming the first case; `stats.json` has the counts.

| tag | meaning | where to look |
| --- | --- | --- |
| `[credentials]` | A key is missing or was rejected: the judge keys (`JUDGE_BASE_URL`, `JUDGE_API_KEY`) or the MIMIC tools (`MIMIC_API_SECRET`) | CliniCARE's `.env` |
| `[agent]` | The model's own behaviour: the agent raised an error, or finished without a report | the trial's `agent/` dir and `trial.log` |
| `[judge]` | An LLM judge failed for a reason other than its key (rate limit, timeout, malformed reply), or a replayed job was never graded | the trial's `verifier/process.json` or `policy_support.json`, or the case's job log |
| `[task build]` | The built task is incomplete: no usable rubric bundle | the trial's `verifier/`; rebuild the tasks |
| `[environment]` | The container, harness or tools failed: no verifier result, the verifier could not start its judge, or the agent reported a degraded environment | the trial's `trial.log`, `result.json` and `verifier/test-stdout.txt` |
| `[timeout]` | The case ran past `case_timeout_sec` and was stopped | the case's job log under `jobs/medhelm/` |

## Run-spec arguments

| argument | default | notes |
| --- | --- | --- |
| `clinicare_root` | `$CLINICARE_ROOT` | CliniCARE checkout |
| `tasks_dir` | `$CLINICARE_TASKS_DIR`, else `<root>/benchmark/tasks` | built tasks; the instance list (coverage denominator) |
| `task_ids` | all | `+`-separated subset |
| `job_dir` | — | replay this finished job dir instead of running |
| `jobs_dir` | `<root>/jobs/medhelm` | where live cases are written |
| `effort` | model map | `minimal` \| `low` \| `medium` \| `high` \| `xhigh` |
| `grade` | `false` | run the LLM judges after each case |
| `case_timeout_sec` | `7200` | per-case ceiling, including harness setup |
| `cohort_csv` | — | `full750.csv`, to attach gold verdicts as references |

## Metrics

| metric | meaning |
| --- | --- |
| `clinicare_score` | **main metric**: fraction of cases with the correct verdict |
| `clinicare_verdict_parse_rate` | report has exactly one parseable verdict |
| `clinicare_grounding_precision` | cited record IDs that were actually retrieved (only cases with a trusted ID pool) |
| `clinicare_policy_doc_f1` | cited guideline documents vs expected (only cases that cited one) |
| `clinicare_tool_calls` | MIMIC tool calls per episode |
| `clinicare_process_score`, `clinicare_policy_support` | LLM-judged, graded runs only |
| `clinicare_trial_failed` | *(not on the leaderboard)* cases with no clean scored trial |
| `clinicare_grading_failed` | *(not on the leaderboard; only with `grade=true`)* cases left without judged scores |
| `clinicare_evidence_calls_auth_error`, `clinicare_environment_degraded` | *(not on the leaderboard)* non-zero means the run lost MIMIC access or a degraded environment, so its numbers are invalid |
| `clinicare_report_present` | *(not on the leaderboard)* the agent wrote a report |

A key a trial did not measure is **omitted, never counted as 0**. Absence varies a lot by system, so zero-filling would change rankings. Efficiency metrics are deliberately not shown: they would count the seed prompt, not the agent's real token spend.

## Privacy and publishing

CliniCARE task ids are a hash of case parameters and **identify the patient** to anyone holding the CliniCARE repo, and reports quote the record. Publish only `run_spec.json`, `scenario.json` and `stats.json`. Use the `rsync --exclude` recipe in `LEADERBOARD_EXPORT.md`, which drops `scenario_state.json`, `per_instance_stats.json` and the files derived from them. Before publishing, check:

- `clinicare_evidence_calls_auth_error` and `clinicare_environment_degraded` are 0;
- `run_spec.json` contains no local paths (set `CLINICARE_ROOT` in the environment instead of on the run entry);
- `grep -r` of the export tree for a known task id finds nothing.

## Troubleshooting

| symptom | fix |
| --- | --- |
| `run_case refused: ... NOT in the allowed_hosts` | A `*_BASE_URL` changed after the tasks were built. Rebuild the tasks with it set. |
| `has no scripts/run_case.py` | Update the CliniCARE checkout. |
| `No CliniCARE system for model ...` | Add the model to `clinicare_model_map.yaml`. |
| Every case scores 0 with zero tool calls | The MIMIC sidecar did not start. Check `MIMIC_DATA_ROOT` and `MIMIC_API_SECRET` in CliniCARE's `.env`. |
| A `[credentials]`, `[agent]`, `[judge]`, `[task build]`, `[environment]` or `[timeout]` warning | See [Warnings](#warnings). |
| Codex trials die with `403 Forbidden` from a gateway | The gateway does not accept the Codex client. Use a gateway that does, or a different system. |
| Codex `invalid_encrypted_content` on the second turn | A load-balancing gateway split the model across deployments. Set `CODEX_PIN_DEPLOYMENT` in CliniCARE's `.env` (see `codex_mcp_fix.py`). |
| Replay: `... built from a different version of their task than ...` | The tasks were rebuilt (e.g. after a CliniCARE update) since the job ran. Point `tasks_dir=` at the build the job used, or re-run live. |
| A re-run never retries a dead case | Harbor keeps finished trials, failed ones included. Delete that case's job dir under `jobs/medhelm/` to re-run it. |
| `run_case failed ... built from a different version of task` or `harbor refused the job` | The jobs dir holds results from an older CliniCARE or task build. Use a fresh `jobs_dir=` (or delete the old one) and re-run. |
| Cases fail with 401s although CliniCARE's `.env` has the right key | A variable exported in your shell wins over `.env`. Unset `OPENAI_API_KEY` / `*_BASE_URL` in the shell that runs `medhelm-run`, or make them match. |

## Summary

| step | command |
| --- | --- |
| CliniCARE setup (once) | `uv sync`, `.env`, `build_image.sh`, `build_tasks.py` in `$CLINICARE_ROOT` |
| Run | `medhelm-run --run-entries "clinicare:model=<m>,model_deployment=<m>" --suite <s>` |
| Replay a finished job | add `job_dir=<job dir>` |
| Graded run | add `grade=true` |
| Leaderboard | `helm-summarize --suite <s>`, then `helm-server --suite <s>` |
