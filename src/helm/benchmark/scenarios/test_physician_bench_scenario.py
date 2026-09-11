import json
from pathlib import Path
from tempfile import TemporaryDirectory

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.physician_bench_adapter import build_pb_request
from helm.benchmark.scenarios.physician_bench_constants import PB_HARNESS_DEPLOYMENT, PB_PROTOCOL
from helm.benchmark.scenarios.physician_bench_scenario import PhysicianBenchScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT, Input, Instance


def _write_fixture(root: Path, task_id: str = "aortic_aneurysm_cad") -> Path:
    task_dir = root / "tasks" / "v1" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(
        f"# {task_id.replace('_', ' ').title()}\n\nDo the clinical task.\n",
        encoding="utf-8",
    )
    return task_dir


def test_physician_bench_scenario_get_instances_parses_envelope_and_extra_data():
    with TemporaryDirectory() as tmpdir:
        pb_root = Path(tmpdir)
        _write_fixture(pb_root)
        scenario = PhysicianBenchScenario(
            pb_root=str(pb_root),
            task_ids="aortic_aneurysm_cad",
            max_steps="30",
            fhir_image="fhir-full:v1",
            port="18080",
        )
        instances = scenario.get_instances(str(pb_root / "out"))

    assert len(instances) == 1
    instance = instances[0]
    assert instance.id == "aortic_aneurysm_cad"
    assert instance.split == TEST_SPLIT
    envelope = json.loads(instance.input.text)
    assert envelope["pb_protocol"] == PB_PROTOCOL
    assert envelope["task_id"] == "aortic_aneurysm_cad"
    assert envelope["task_relpath"] == "tasks/v1/aortic_aneurysm_cad"
    assert envelope["goal"] == "Aortic Aneurysm Cad"
    assert envelope["max_steps"] == 30
    assert envelope["fhir_image"] == "fhir-full:v1"
    assert envelope["port"] == 18080
    assert instance.extra_data is not None
    assert instance.extra_data["task_relpath"] == "tasks/v1/aortic_aneurysm_cad"


def test_physician_bench_scenario_one_instance_per_task():
    """Instances are task folders; pytest checkpoints stay inside the episode score."""
    with TemporaryDirectory() as tmpdir:
        pb_root = Path(tmpdir)
        _write_fixture(pb_root, "aortic_aneurysm_cad")
        _write_fixture(pb_root, "chronic_cough_geriatric")
        scenario = PhysicianBenchScenario(pb_root=str(pb_root))
        instances = scenario.get_instances(str(pb_root / "out"))

    assert [instance.id for instance in instances] == [
        "aortic_aneurysm_cad",
        "chronic_cough_geriatric",
    ]


def test_physician_bench_scenario_filters_task_ids():
    with TemporaryDirectory() as tmpdir:
        pb_root = Path(tmpdir)
        _write_fixture(pb_root, "aortic_aneurysm_cad")
        _write_fixture(pb_root, "chronic_cough_geriatric")
        scenario = PhysicianBenchScenario(
            pb_root=str(pb_root),
            task_ids="aortic_aneurysm_cad",
        )
        instances = scenario.get_instances(str(pb_root / "out"))
    assert [instance.id for instance in instances] == ["aortic_aneurysm_cad"]


def test_lookup_model_mapping_exact_prefix_and_fail_closed():
    from helm.clients.physician_bench_client import lookup_model_mapping

    stub = lookup_model_mapping("simple/model1")
    assert stub is not None
    assert stub["backend"] == "stub"
    openai = lookup_model_mapping("openai/gpt-4o-2024-05-13")
    assert openai is not None
    assert openai["backend"] == "physician_bench"
    claude = lookup_model_mapping("anthropic/claude-opus-4.7")
    assert claude is not None
    assert lookup_model_mapping("unknown/model") is None
    mini = lookup_model_mapping("openai/gpt-5-mini")
    assert mini is not None
    assert mini["backend"] == "physician_bench"
    assert mini["pb_model_id"] == "openai/gpt-5-mini"
    dated = lookup_model_mapping("openai/gpt-5-mini-2025-08-07")
    assert dated is not None
    assert dated["pb_model_id"] == "openai/gpt-5-mini"


def test_build_pb_request_rewrites_model_deployment_and_injects_evaluated_model():
    envelope = {
        "pb_protocol": PB_PROTOCOL,
        "task_id": "aortic_aneurysm_cad",
        "task_relpath": "tasks/v1/aortic_aneurysm_cad",
        "goal": "do the task",
    }
    instance = Instance(
        input=Input(text=json.dumps(envelope)),
        references=[],
        split=TEST_SPLIT,
        id="aortic_aneurysm_cad",
    )
    adapter_spec = AdapterSpec(
        method="physician_bench",
        model="openai/gpt-4o-2024-05-13",
        model_deployment="openai/gpt-4o-2024-05-13",
        instructions=json.dumps({"max_steps": 30, "port": 18080}),
        max_train_instances=0,
        max_tokens=1,
        temperature=0.0,
        num_outputs=1,
    )
    request = build_pb_request(instance, adapter_spec)
    assert request.model_deployment == PB_HARNESS_DEPLOYMENT
    assert request.model == "openai/gpt-4o-2024-05-13"
    payload = json.loads(request.prompt)
    assert payload["evaluated_model"] == "openai/gpt-4o-2024-05-13"
    assert payload["evaluated_model_deployment"] == "openai/gpt-4o-2024-05-13"
    assert payload["max_steps"] == 30
    assert payload["port"] == 18080


def test_physician_bench_spec_wires_adapter_and_metric():
    from helm.benchmark.run_specs.medhelm_run_specs import get_physician_bench_spec

    spec = get_physician_bench_spec(task_ids="aortic_aneurysm_cad", max_steps="30")
    assert spec.groups == ["physician_bench"]
    assert spec.adapter_spec.method == "physician_bench"
    knobs = json.loads(spec.adapter_spec.instructions)
    assert knobs["max_steps"] == 30
    assert spec.scenario_spec.args["task_ids"] == "aortic_aneurysm_cad"
    metric_classes = [metric.class_name for metric in spec.metric_specs]
    assert "helm.benchmark.metrics.physician_bench_metrics.PhysicianBenchMetric" in metric_classes


def test_physician_bench_accepts_openrouter_gpt5_mini_id():
    from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
    from helm.benchmark.run_spec_factory import construct_run_specs
    from helm.common.object_spec import parse_object_spec

    register_builtin_configs_from_helm_package()
    specs = construct_run_specs(
        parse_object_spec(
            "physician_bench:task_ids=aortic_aneurysm_cad,"
            "model=openai/gpt-5-mini,model_deployment=openai/gpt-5-mini"
        )
    )
    assert len(specs) == 1
    assert specs[0].adapter_spec.model == "openai/gpt-5-mini"
    assert specs[0].adapter_spec.model_deployment == "openai/gpt-5-mini"
    assert specs[0].adapter_spec.method == "physician_bench"
