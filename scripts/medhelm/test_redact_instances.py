import json
import os
from typing import List

from helm.benchmark.adaptation.adapter_spec import ADAPT_MULTIPLE_CHOICE_JOINT, AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
)
from helm.common.codec import to_json
from helm.common.request import GeneratedOutput, Request, RequestResult, Thinking

from redact_instances import (
    REDACTED_STRING,
    SCENARIO_STATE_FILE_NAME,
    check_structure,
    get_scenario_name,
    modify_scenario_states_for_suite,
    redact_request_state,
)


PASSAGE = "Case presentation: a synthetic patient with a synthetic complaint."
QUESTION = "What is the synthetic diagnosis?"
OPTIONS = {"A": "Synthetic option one", "B": "Synthetic option two"}


def _build_request_state() -> RequestState:
    references: List[Reference] = [
        Reference(Output(text=OPTIONS[letter]), tags=[CORRECT_TAG] if letter == "B" else [])
        for letter in sorted(OPTIONS)
    ]
    instance = Instance(
        input=Input(text=f"{PASSAGE}\n\n{QUESTION}"),
        references=references,
        split=TEST_SPLIT,
        sub_split="synthetic",
        id="id42",
    )
    completion = GeneratedOutput(
        text="B",
        logprob=-0.5,
        tokens=[],
        thinking=Thinking(text=f"The case said: {PASSAGE}"),
    )
    return RequestState(
        instance=instance,
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping=dict(OPTIONS),
        request=Request(model="simple/model1", model_deployment="simple/model1", prompt=f"{PASSAGE}\n\n{QUESTION}"),
        result=RequestResult(success=True, embedding=[], completions=[completion], cached=False),
        num_train_instances=0,
        prompt_truncated=False,
    )


def _build_scenario_state() -> ScenarioState:
    return ScenarioState(
        adapter_spec=AdapterSpec(method=ADAPT_MULTIPLE_CHOICE_JOINT),
        request_states=[_build_request_state()],
    )


def test_redact_request_state_redacts_licensed_text():
    redacted = redact_request_state(_build_request_state())

    assert redacted.instance.input.text == REDACTED_STRING
    assert [reference.output.text for reference in redacted.instance.references] == [REDACTED_STRING] * 2
    assert redacted.request.prompt == REDACTED_STRING
    assert redacted.output_mapping is None
    assert redacted.result is not None
    assert redacted.result.completions[0].thinking is None


def test_redact_request_state_preserves_structure():
    redacted = redact_request_state(_build_request_state())

    assert redacted.instance.id == "id42"
    assert redacted.instance.split == TEST_SPLIT
    assert redacted.instance.sub_split == "synthetic"
    assert redacted.instance.perturbation is None
    assert [reference.tags for reference in redacted.instance.references] == [[], [CORRECT_TAG]]
    assert redacted.result is not None
    assert redacted.result.completions[0].text == "B"


def test_redact_request_state_removes_every_licensed_string():
    serialized = to_json(redact_request_state(_build_request_state()))

    for licensed in [PASSAGE, QUESTION, *OPTIONS.values()]:
        assert licensed not in serialized


def test_get_scenario_name():
    assert get_scenario_name("cpc_qa:dataset=dataset_100,model=openai_gpt2") == "cpc_qa"
    assert get_scenario_name("cpc_qa") == "cpc_qa"
    assert get_scenario_name("pubmed_qa:model=openai_gpt2") == "pubmed_qa"


def test_check_structure_fails_before_redaction_and_passes_after():
    scenario_state = _build_scenario_state()
    assert check_structure(scenario_state)

    redacted = ScenarioState(
        adapter_spec=scenario_state.adapter_spec,
        request_states=[redact_request_state(request_state) for request_state in scenario_state.request_states],
    )
    assert check_structure(redacted) == []


def _write_run(run_suite_path: str, run_dir_name: str) -> str:
    scenario_state_path = os.path.join(run_suite_path, run_dir_name, SCENARIO_STATE_FILE_NAME)
    os.makedirs(os.path.dirname(scenario_state_path))
    with open(scenario_state_path, "w") as f:
        f.write(to_json(_build_scenario_state()))
    return scenario_state_path


def test_modify_scenario_states_for_suite_only_touches_restricted_scenarios(tmp_path):
    run_suite_path = str(tmp_path)
    restricted_path = _write_run(run_suite_path, "cpc_qa:model=simple_model1")
    other_path = _write_run(run_suite_path, "pubmed_qa:model=simple_model1")
    other_contents_before = open(other_path).read()

    modify_scenario_states_for_suite(run_suite_path, {"cpc_qa"})

    with open(restricted_path) as f:
        restricted = json.load(f)
    assert restricted["request_states"][0]["instance"]["input"]["text"] == REDACTED_STRING
    assert "output_mapping" not in restricted["request_states"][0]

    assert open(other_path).read() == other_contents_before
