"""Removes licensed scenario content from run artifacts

This script modifies the scenario_state.json files of restricted scenarios in place
within a suite, removing the instance input, the answer options and the request prompt
from every `RequestState`.

This is used when a scenario is built from licensed data that must not be redistributed
(e.g. CPC-QA, whose NEJM case records are downloaded separately and passed to the
scenario via `data_path`). Run artifacts otherwise contain the full case text, and
`helm-server` serves everything under the output path with no access check.

Metrics are computed and written before this script runs, so no score can change.

After running this, you must re-run helm-summarize on the suite in order to
update other JSON files used by the web frontend. Then re-run this script with
`--check` to confirm that no licensed content survived:

    python scripts/medhelm/redact_instances.py --suite <suite>
    helm-summarize --suite <suite>
    python scripts/medhelm/redact_instances.py --suite <suite> --check
"""

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Set

from tqdm import tqdm

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.scenarios.scenario import Input, Output
from helm.common.codec import from_json, to_json
from helm.common.hierarchical_logger import hwarn
from helm.common.request import RequestResult


SCENARIO_STATE_FILE_NAME = "scenario_state.json"
RUN_SPEC_FILE_NAME = "run_spec.json"
REDACTED_STRING = "[redacted]"

RESTRICTED_SCENARIOS = {"cpc_qa"}
"""Scenarios whose instances contain licensed content. Overridable with --scenarios."""

LICENSED_RECORD_FIELDS = ["presentation_of_case", "question"]
"""Dataset record fields holding licensed free text, alongside the answer options."""

MIN_CHECKED_SUBSTRING_LENGTH = 8
"""Licensed strings shorter than this are too short to search for without false positives."""

MAX_PLAUSIBLE_COMPLETION_LENGTH = 200
"""Completions longer than this are warned about: a reasoning trace delivered as ordinary
completion text could quote the case back, and completion text is deliberately preserved."""


def read_scenario_state(scenario_state_path: str) -> ScenarioState:
    if not os.path.exists(scenario_state_path):
        raise ValueError(f"Could not load ScenarioState from {scenario_state_path}")
    with open(scenario_state_path) as f:
        return from_json(f.read(), ScenarioState)


def write_scenario_state(scenario_state_path: str, scenario_state: ScenarioState) -> None:
    with open(scenario_state_path, "w") as f:
        f.write(to_json(scenario_state))


def get_scenario_name(run_dir_name: str) -> str:
    """Return the scenario name from a run directory name e.g. "cpc_qa:model=openai_gpt2"."""
    return run_dir_name.split(":")[0]


def redact_result(result: RequestResult) -> RequestResult:
    """Drop reasoning traces, keeping completion text so model answers stay debuggable."""
    completions = [dataclasses.replace(completion, thinking=None) for completion in result.completions]
    return dataclasses.replace(result, completions=completions)


def redact_request_state(request_state: RequestState) -> RequestState:
    """Blank out licensed text while keeping the structure the frontend relies on.

    `instance.id`, `split`, `sub_split` and `perturbation` are preserved because the
    instance id is what joins display_predictions.json to instances.json. Reference tags
    are preserved so that CORRECT_TAG survives.
    """
    references = [
        dataclasses.replace(reference, output=Output(text=REDACTED_STRING))
        for reference in request_state.instance.references
    ]
    instance = dataclasses.replace(request_state.instance, input=Input(text=REDACTED_STRING), references=references)
    return dataclasses.replace(
        request_state,
        instance=instance,
        # Otherwise mapped_output leaks the option text into display_predictions.json.
        output_mapping=None,
        request=dataclasses.replace(request_state.request, prompt=REDACTED_STRING),
        result=redact_result(request_state.result) if request_state.result is not None else None,
    )


def redact_scenario_state(scenario_state: ScenarioState) -> ScenarioState:
    redacted_request_states = [redact_request_state(request_state) for request_state in scenario_state.request_states]
    return dataclasses.replace(scenario_state, request_states=redacted_request_states)


def modify_scenario_state_for_run(run_path: str) -> None:
    scenario_state_path = os.path.join(run_path, SCENARIO_STATE_FILE_NAME)
    scenario_state = read_scenario_state(scenario_state_path)
    redacted_scenario_state = redact_scenario_state(scenario_state)
    write_scenario_state(scenario_state_path, redacted_scenario_state)


def read_data_path(run_path: str) -> Optional[str]:
    """Return the scenario's `data_path` argument as recorded in run_spec.json."""
    run_spec_path = os.path.join(run_path, RUN_SPEC_FILE_NAME)
    if not os.path.exists(run_spec_path):
        return None
    with open(run_spec_path) as f:
        run_spec = json.load(f)
    data_path = run_spec.get("scenario_spec", {}).get("args", {}).get("data_path")
    return data_path if isinstance(data_path, str) else None


def read_licensed_substrings(data_path: str) -> Set[str]:
    """Collect the licensed strings of the source dataset that must not appear in artifacts.

    Strings shorter than `MIN_CHECKED_SUBSTRING_LENGTH` are dropped, because short option
    texts match unrelated artifact content.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    substrings: Set[str] = set()
    for record in records:
        for field_name in LICENSED_RECORD_FIELDS:
            value = record.get(field_name)
            if isinstance(value, str):
                substrings.add(value)
        for option in record.get("options", {}).values():
            if isinstance(option, str):
                substrings.add(option)

    long_enough = {substring for substring in substrings if len(substring) >= MIN_CHECKED_SUBSTRING_LENGTH}
    skipped = len(substrings) - len(long_enough)
    if skipped:
        hwarn(f"{skipped} licensed strings are shorter than {MIN_CHECKED_SUBSTRING_LENGTH} characters, not searched")
    return long_enough


def _iter_strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _iter_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_strings(value)


def read_artifact_text(file_path: str) -> str:
    """Return the searchable text of a run artifact.

    JSON files are decoded first, so that escape sequences in the file do not hide a
    literal match.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        contents = f.read()
    try:
        parsed = json.loads(contents)
    except json.JSONDecodeError:
        return contents
    return "\n".join(_iter_strings(parsed))


def check_content(run_path: str, licensed_substrings: Set[str]) -> List[str]:
    """Assert that no licensed string of the source dataset appears in any run artifact."""
    failures: List[str] = []
    file_names = sorted(p for p in os.listdir(run_path) if os.path.isfile(os.path.join(run_path, p)))
    for file_name in file_names:
        artifact_text = read_artifact_text(os.path.join(run_path, file_name))
        leaked = [substring for substring in licensed_substrings if substring in artifact_text]
        if leaked:
            failures.append(f"{file_name} contains {len(leaked)} licensed string(s), e.g. {leaked[0][:80]!r}")
    return failures


def check_structure(scenario_state: ScenarioState) -> List[str]:
    """Assert that every request state was redacted, without reading the source dataset."""
    failures: List[str] = []
    for index, request_state in enumerate(scenario_state.request_states):
        instance_id = request_state.instance.id or f"index {index}"
        if request_state.instance.input.text != REDACTED_STRING:
            failures.append(f"instance {instance_id} has an unredacted input")
        for reference in request_state.instance.references:
            if reference.output.text != REDACTED_STRING:
                failures.append(f"instance {instance_id} has an unredacted reference")
                break
        if request_state.request.prompt != REDACTED_STRING:
            failures.append(f"instance {instance_id} has an unredacted prompt")
        if request_state.output_mapping is not None:
            failures.append(f"instance {instance_id} has an output_mapping")
    return failures


def warn_on_long_completions(run_dir_name: str, scenario_state: ScenarioState) -> None:
    for request_state in scenario_state.request_states:
        if request_state.result is None:
            continue
        for completion in request_state.result.completions:
            if len(completion.text) > MAX_PLAUSIBLE_COMPLETION_LENGTH:
                hwarn(
                    f"{run_dir_name}: instance {request_state.instance.id} has a "
                    f"{len(completion.text)}-character completion, which may quote the case"
                )
                return


def check_run(run_path: str, run_dir_name: str) -> List[str]:
    scenario_state = read_scenario_state(os.path.join(run_path, SCENARIO_STATE_FILE_NAME))
    failures = check_structure(scenario_state)
    warn_on_long_completions(run_dir_name, scenario_state)

    data_path = read_data_path(run_path)
    if data_path is None or not os.path.exists(data_path):
        hwarn(f"{run_dir_name}: source dataset {data_path} is unreadable, checking structure only")
        return failures

    return failures + check_content(run_path, read_licensed_substrings(data_path))


def get_restricted_run_dir_names(run_suite_path: str, scenarios: Set[str]) -> List[str]:
    """Return the run directories of restricted scenarios, in sorted order."""
    # run_suite_path can contain subdirectories that are not runs (e.g. eval_cache, groups)
    # so filter them out.
    return sorted(
        [
            p
            for p in os.listdir(run_suite_path)
            if p != "eval_cache"
            and p != "groups"
            and os.path.isdir(os.path.join(run_suite_path, p))
            and get_scenario_name(p) in scenarios
        ]
    )


def modify_scenario_states_for_suite(run_suite_path: str, scenarios: Set[str]) -> None:
    run_dir_names = get_restricted_run_dir_names(run_suite_path, scenarios)
    for run_dir_name in tqdm(run_dir_names, disable=None):
        run_path: str = os.path.join(run_suite_path, run_dir_name)
        if not os.path.exists(os.path.join(run_path, SCENARIO_STATE_FILE_NAME)):
            hwarn(f"{run_dir_name} doesn't have {SCENARIO_STATE_FILE_NAME}, skipping")
            continue
        modify_scenario_state_for_run(run_path)


def check_scenario_states_for_suite(run_suite_path: str, scenarios: Set[str]) -> Dict[str, List[str]]:
    """Return the check failures of each restricted run, keyed by run directory name."""
    failures_by_run: Dict[str, List[str]] = {}
    for run_dir_name in tqdm(get_restricted_run_dir_names(run_suite_path, scenarios), disable=None):
        run_path: str = os.path.join(run_suite_path, run_dir_name)
        if not os.path.exists(os.path.join(run_path, SCENARIO_STATE_FILE_NAME)):
            hwarn(f"{run_dir_name} doesn't have {SCENARIO_STATE_FILE_NAME}, skipping")
            continue
        failures = check_run(run_path, run_dir_name)
        if failures:
            failures_by_run[run_dir_name] = failures
    return failures_by_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Name of the suite to redact.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=sorted(RESTRICTED_SCENARIOS),
        help="Names of the scenarios to redact.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the suite is already redacted instead of redacting it. "
        "Does not modify any file and exits non-zero if any check fails.",
    )
    args = parser.parse_args()
    run_suite_path = os.path.join(args.output_path, "runs", args.suite)
    scenarios = set(args.scenarios)

    if not args.check:
        modify_scenario_states_for_suite(run_suite_path, scenarios)
        return

    failures_by_run = check_scenario_states_for_suite(run_suite_path, scenarios)
    for run_dir_name, failures in failures_by_run.items():
        for failure in failures:
            print(f"FAIL {run_dir_name}: {failure}", file=sys.stderr)
    if failures_by_run:
        sys.exit(f"{len(failures_by_run)} run(s) in {run_suite_path} contain licensed content")
    print(f"All restricted runs in {run_suite_path} are redacted")


if __name__ == "__main__":
    main()
