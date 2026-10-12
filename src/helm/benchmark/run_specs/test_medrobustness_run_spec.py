import collections
from pathlib import Path

import pandas as pd

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION
from helm.benchmark.presentation.run_entry import RunEntry
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.runner import downsample_eval_instances
from helm.benchmark.run_specs.medhelm_run_specs import get_medrobustness_format_spec
from helm.benchmark.scenarios.scenario import create_scenario, with_instance_ids


def _write_workbook(path: Path, case_count: int) -> None:
    rows = pd.DataFrame(
        [
            {
                "id": f"FOR-{index:03d}",
                "benchmark": "PR26-Format",
                "core_id": f"S{index:02d}",
                "scoring_mode": "preserve",
                "family": "table",
                "transformation": "prose record to table",
                "arm_a": f"Baseline prompt {index}",
                "arm_b": f"Formatted prompt {index}",
                "correct_letter_a": "A",
                "correct_letter_b": "B",
                "ascertain_letter": "F",
                "cannot_decide_letter": "G",
            }
            for index in range(1, case_count + 1)
        ]
    )
    with pd.ExcelWriter(path) as writer:
        rows.to_excel(writer, sheet_name="Test cases", index=False)


def test_medrobustness_format_run_spec_uses_verbatim_zero_shot_prompts() -> None:
    data_path = "/data/PR26-Format-g1.xlsx"

    run_spec = get_medrobustness_format_spec(data_path)

    assert run_spec.name == "medrobustness_format"
    assert run_spec.scenario_spec.args == {"data_path": data_path}
    assert run_spec.adapter_spec.method == ADAPT_GENERATION
    assert run_spec.adapter_spec.instructions == ""
    assert run_spec.adapter_spec.input_prefix == ""
    assert run_spec.adapter_spec.input_suffix == ""
    assert run_spec.adapter_spec.output_prefix == ""
    assert run_spec.adapter_spec.output_suffix == ""
    assert run_spec.adapter_spec.instance_prefix == ""
    assert run_spec.adapter_spec.max_train_instances == 0
    assert run_spec.adapter_spec.num_outputs == 1
    assert run_spec.adapter_spec.temperature == 0.0
    assert run_spec.metric_specs[0].class_name == ("helm.benchmark.metrics.medrobustness_metrics.MedRobustnessMetric")


def test_cli_cap_reaches_scenario_and_keeps_every_pair_intact(tmp_path: Path) -> None:
    data_path = tmp_path / "PR26-Format-g1.xlsx"
    _write_workbook(data_path, case_count=20)
    run_entry = RunEntry(description=f"medrobustness_format:data_path={data_path},model=simple/model1", priority=1)

    run_spec = run_entries_to_run_specs([run_entry], max_eval_instances=40)[0]
    assert run_spec.scenario_spec.args["max_eval_instances"] == 40
    instances = with_instance_ids(create_scenario(run_spec.scenario_spec).get_instances(str(tmp_path)))
    sampled = downsample_eval_instances(instances, 40, ["test"])

    pair_counts = collections.Counter(instance.extra_data["pair_id"] for instance in sampled)  # type: ignore[index]
    assert len(sampled) == len(instances) == 40
    assert set(pair_counts.values()) == {2}
