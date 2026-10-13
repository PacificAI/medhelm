from pathlib import Path
from typing import List, Type

import pandas as pd
import pytest

from helm.benchmark.scenarios.medrobustness.pr26_base_scenario import PR26BaseScenario
from helm.benchmark.scenarios.medrobustness.pr26_scenarios import PR26ControlScenario
from helm.benchmark.scenarios.medrobustness.pr26_scenarios import PR26FormatScenario
from helm.benchmark.scenarios.medrobustness.pr26_scenarios import PR26NegationScenario
from helm.benchmark.scenarios.medrobustness.pr26_scenarios import PR26NumeracyScenario
from helm.benchmark.scenarios.medrobustness.pr26_scenarios import PR26ParaphraseScenario
from helm.benchmark.scenarios.medrobustness.pr26_scenarios import PR26PositionScenario
from helm.benchmark.scenarios.medrobustness.pr26_scenarios import PR26TemporalScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT

SCENARIOS: List[Type[PR26BaseScenario]] = [
    PR26FormatScenario,
    PR26ControlScenario,
    PR26PositionScenario,
    PR26ParaphraseScenario,
    PR26NegationScenario,
    PR26TemporalScenario,
    PR26NumeracyScenario,
]


def _write_workbook(path: Path, case_ids: List[str]) -> None:
    rows = pd.DataFrame(
        [
            {
                "id": case_id,
                "benchmark": "PR26-Test",
                "core_id": "S01",
                "scoring_mode": "preserve",
                "family": "family",
                "transformation": "transformation",
                "arm_a": f"Baseline {case_id}",
                "arm_b": f"Variant {case_id}",
                "correct_letter_a": "A",
                "correct_letter_b": "B",
                "ascertain_letter": "F",
                "cannot_decide_letter": "G",
            }
            for case_id in case_ids
        ]
    )
    with pd.ExcelWriter(path) as writer:
        rows.to_excel(writer, sheet_name="Test cases", index=False)


@pytest.mark.parametrize("scenario_class", SCENARIOS)
def test_each_scenario_emits_whole_pairs_and_respects_max_eval_instances(
    tmp_path: Path, scenario_class: Type[PR26BaseScenario]
) -> None:
    data_path = tmp_path / "workbook.xlsx"
    _write_workbook(data_path, ["TST-001", "TST-002"])

    instances = scenario_class(str(data_path)).get_instances("unused")
    assert len(instances) == 4
    assert {instance.split for instance in instances} == {TEST_SPLIT}
    pairs = {(i.extra_data["pair_id"], i.extra_data["arm"]) for i in instances if i.extra_data}  # type: ignore[index]
    assert pairs == {("TST-001", "A"), ("TST-001", "B"), ("TST-002", "A"), ("TST-002", "B")}

    capped = scenario_class(str(data_path), max_eval_instances=2).get_instances("unused")
    assert [instance.extra_data["arm"] for instance in capped if instance.extra_data] == ["A", "B"]  # type: ignore[index]


def test_format_scenario_preserves_both_paired_arms(tmp_path: Path) -> None:
    data_path = tmp_path / "Format.xlsx"
    _write_workbook(data_path, ["FOR-001"])

    instances = PR26FormatScenario(str(data_path)).get_instances("unused")

    assert [instance.id for instance in instances] == ["FOR-001:A", "FOR-001:B"]
    assert [instance.input.text for instance in instances] == ["Baseline FOR-001", "Variant FOR-001"]
    assert [instance.first_correct_reference.output.text for instance in instances] == ["A", "B"]  # type: ignore[union-attr]


def test_format_scenario_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "Format.xlsx"
    _write_workbook(data_path, ["FOR-001", "FOR-001"])

    with pytest.raises(ValueError, match="Duplicate case IDs: FOR-001"):
        PR26FormatScenario(str(data_path)).get_instances("unused")


def test_control_ranks_on_noise_floor_and_others_on_fully_correct_rate() -> None:
    assert PR26ControlScenario(data_path="unused.xlsx").get_metadata().main_metric == (
        "medrobustness_control_failure_rate"
    )
    for scenario_class in SCENARIOS:
        if scenario_class is not PR26ControlScenario:
            metadata = scenario_class(data_path="unused.xlsx").get_metadata()
            assert metadata.name == scenario_class.name
            assert metadata.main_metric == "medrobustness_fully_correct_rate"
