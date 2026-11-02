import json
from dataclasses import replace

from helm.benchmark.adaptation.adapters.clinicare_adapter import build_clinicare_request
from helm.benchmark.run_specs.medhelm_run_specs import get_clinicare_spec
from helm.benchmark.scenarios.clinicare_constants import CLINICARE_HARNESS_DEPLOYMENT, CLINICARE_PROTOCOL
from helm.benchmark.scenarios.scenario import TEST_SPLIT, Input, Instance

TASK = "advanced-imaging-000001"  # placeholder: real task ids identify patients


def _instance() -> Instance:
    return Instance(
        input=Input(text="placeholder prompt"),
        references=[],
        split=TEST_SPLIT,
        id=TASK,
        extra_data={"task_id": TASK, "scenario_id": "advanced-imaging"},
    )


def test_request_keeps_evaluated_model_and_reroutes_deployment():
    spec = get_clinicare_spec(clinicare_root="/opt/clinicare", effort="high", grade="true")
    adapter_spec = replace(spec.adapter_spec, model="openai/gpt-x", model_deployment="openai/gpt-x")
    request = build_clinicare_request(_instance(), adapter_spec)

    assert request.model == "openai/gpt-x"
    assert request.model_deployment == CLINICARE_HARNESS_DEPLOYMENT
    envelope = json.loads(request.prompt)
    assert envelope["clinicare_protocol"] == CLINICARE_PROTOCOL
    assert envelope["task_id"] == TASK
    assert envelope["evaluated_model"] == "openai/gpt-x"
    assert envelope["clinicare_root"] == "/opt/clinicare"
    assert envelope["effort"] == "high"
    assert envelope["grade"] is True
    assert "job_dir" not in envelope  # empty knobs are dropped, so live mode is the default


def test_replay_knob_reaches_envelope():
    spec = get_clinicare_spec(tasks_dir="/t", job_dir="/j")
    request = build_clinicare_request(_instance(), replace(spec.adapter_spec, model="openai/gpt-x"))
    envelope = json.loads(request.prompt)
    assert envelope["job_dir"] == "/j" and envelope["tasks_dir"] == "/t"
    assert "grade" not in envelope


def test_run_spec_has_outcome_and_offline_metrics():
    classes = [m.class_name for m in get_clinicare_spec().metric_specs]
    assert "helm.benchmark.metrics.clinicare_metrics.CliniCAREOutcomeMetric" in classes
    assert "helm.benchmark.metrics.clinicare_metrics.CliniCAREOfflineMetric" in classes


def test_empty_args_stay_out_of_the_scenario_spec():
    # Scenario args are shown in published group titles: a default run entry must carry no paths.
    assert get_clinicare_spec().scenario_spec.args == {}
    assert get_clinicare_spec(task_ids="a+b").scenario_spec.args == {"task_ids": "a+b"}


def test_settings_that_change_the_measurement_get_distinct_run_names():
    # HELM appends only model/model_deployment, so without this an effort or graded run would
    # overwrite the plain run's output dir under runs/<suite>/.
    names = {
        get_clinicare_spec().name,
        get_clinicare_spec(effort="high").name,
        get_clinicare_spec(grade="true").name,
        get_clinicare_spec(effort="high", grade="true").name,
    }
    assert len(names) == 4
    assert get_clinicare_spec().name == "clinicare"
    # Paths and task ids never reach the (published) run name.
    assert get_clinicare_spec(clinicare_root="/x", tasks_dir="/y", task_ids="a+b").name == "clinicare"
