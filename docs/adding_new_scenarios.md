---
title: Adding New Scenarios
---
# Adding New Scenarios

MedHELM ships with [medical evaluation scenarios](/scenarios) built into the repository. You can also add custom scenarios without forking core framework code by placing modules on your `PYTHONPATH`, or contribute new scenarios directly to MedHELM.

There are two steps: implement a `Scenario` subclass, then add a run spec function.

The easiest approach is to copy from an existing MedHELM example and adapt it. Match your **task** to one of the patterns in `simple_scenarios.py` and `simple_run_specs.py`:

- **Multiple-choice QA:** `SimpleMCQAScenario` / `get_simple_mcqa_run_spec()`
- **Short-answer QA:** `SimpleShortAnswerQAScenario` / `get_simple_short_answer_qa_run_spec()`
- **Multi-class classification:** `SimpleClassificationScenario` / `get_simple_classification_run_spec()`

For production MedHELM scenarios, see real implementations such as:

- `pubmed_qa_scenario.py` — Hugging Face dataset, multiple-choice
- `medi_qa_scenario.py` — Hugging Face dataset with retry logic
- `med_dialog_scenario.py` — external data download
- `dischargeme_scenario.py` — summarization (requires `[summarization]` extra)
- `health_admin_bench_scenario.py` — computer-use HealthAdminBench wrap (see [HealthAdminBench](/health_admin_bench))

Run specs for MedHELM live in `src/helm/benchmark/run_specs/medhelm_run_specs.py`.

## Custom `Scenario` subclass

Create `my_scenario.py` with a class extending `Scenario`. Copy structure and imports from a similar existing scenario (e.g. `pubmed_qa_scenario.py`).

Add a test file `test_my_scenario.py` under `src/helm/benchmark/scenarios/`, following patterns in `test_medi_qa_scenario.py` or `test_pubmed_qa_scenario.py`:

```bash
uv run pytest src/helm/benchmark/scenarios/test_my_scenario.py -vv
```

Implement `get_instances()` to load your dataset. Use `output_path` for cached downloads under `benchmark_output/scenarios/`.

### Downloading data to local disk

Use `ensure_directory_exists()` and `ensure_file_downloaded()` from `helm.common.general` so files are cached between runs. Set `unpack=True` for archive files.

Examples in this repository:

- `med_qa_scenario.py` — Google Drive via `gdown` (requires `[gated]`)
- `med_dialog_scenario.py` — CodaLab bundle download
- `medication_qa_scenario.py` — Excel (`.xlsx`) via pandas

### Working with Hugging Face datasets

Use `load_dataset()` with `cache_dir` set to a subdirectory of `output_path` for hermetic caching.

Examples:

- `pubmed_qa_scenario.py`
- `medi_qa_scenario.py`
- `health_bench_scenario.py`

## Custom run spec function

A run spec function returns a `RunSpec` that wires together scenario, adapter, and metrics. HELM discovers functions in:

- `helm.benchmark.run_specs.*_run_specs`
- Root modules matching `helm_*_run_specs`

For MedHELM contributions, add your function to `src/helm/benchmark/run_specs/medhelm_run_specs.py` with the `@run_spec_function("your_name")` decorator. Copy from a similar run spec in that file (e.g. `get_pubmed_qa_spec()`).

Test locally:

```bash
medhelm-run --run-entries your_name:model=openai/gpt2 --suite custom --max-eval-instances 5
```

For fast iteration without model latency, use the echo model:

```bash
medhelm-run --run-entries your_name:model=simple/model1 --suite custom --max-eval-instances 5
```

For custom modules outside the repo, see [Importing Custom Modules](/importing_custom_modules).

## Contributing to MedHELM

We welcome contributions that:

- Evaluate LLMs on realistic medical tasks
- Use publicly available datasets (or document gated/private access clearly)
- Fill gaps in MedHELM coverage

When contributing:

1. Place `your_scenario.py` in `src/helm/benchmark/scenarios/`
2. Place run spec functions in `src/helm/benchmark/run_specs/medhelm_run_specs.py` (or a dedicated `*_run_specs.py` if appropriate)
3. Add `test_your_scenario.py` with at least unit tests; use `@pytest.mark.scenarios` for integration tests that download data
4. Add an entry to `src/helm/benchmark/static/schema_medhelm.yaml` if the scenario should appear on the leaderboard
5. Open a pull request
