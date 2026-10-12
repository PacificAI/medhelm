from pathlib import Path
from typing import Dict, List, Set, Type

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


def _write_workbook(path: Path, benchmark: str, id_prefix: str) -> None:
    rows = pd.DataFrame(
        [
            {
                "id": f"{id_prefix}-{n:03d}",
                "benchmark": benchmark,
                "core_id": f"S{n:02d}",
                "scoring_mode": "preserve",
                "family": "family",
                "transformation": "transformation",
                "arm_a": f"Baseline {n}",
                "arm_b": f"Variant {n}",
                "correct_letter_a": "A",
                "correct_letter_b": "B",
                "ascertain_letter": "F",
                "cannot_decide_letter": "G",
            }
            for n in (1, 2)
        ]
    )
    with pd.ExcelWriter(path) as writer:
        rows.to_excel(writer, sheet_name="Test cases", index=False)


@pytest.mark.parametrize("scenario_class", SCENARIOS)
def test_each_scenario_is_a_pr26_base_scenario(scenario_class: Type[PR26BaseScenario]) -> None:
    assert issubclass(scenario_class, PR26BaseScenario)
    assert scenario_class.name.startswith("pr26_")
    assert scenario_class.display_name.startswith("PacificMedRobustness: ")


@pytest.mark.parametrize("scenario_class", SCENARIOS)
def test_each_scenario_emits_whole_pairs(tmp_path: Path, scenario_class: Type[PR26BaseScenario]) -> None:
    data_path = tmp_path / "workbook.xlsx"
    _write_workbook(data_path, benchmark="PR26-Test", id_prefix="TST")

    instances = scenario_class(str(data_path)).get_instances("unused")

    assert len(instances) == 4
    assert {instance.split for instance in instances} == {TEST_SPLIT}
    pairs: Dict[str, Set[str]] = {}
    for instance in instances:
        assert instance.extra_data is not None
        pairs.setdefault(instance.extra_data["pair_id"], set()).add(instance.extra_data["arm"])
    assert all(arms == {"A", "B"} for arms in pairs.values())


@pytest.mark.parametrize("scenario_class", SCENARIOS)
def test_each_scenario_respects_max_eval_instances(tmp_path: Path, scenario_class: Type[PR26BaseScenario]) -> None:
    data_path = tmp_path / "workbook.xlsx"
    _write_workbook(data_path, benchmark="PR26-Test", id_prefix="TST")

    instances = scenario_class(str(data_path), max_eval_instances=2).get_instances("unused")

    assert len(instances) == 2
    assert {instance.extra_data["arm"] for instance in instances if instance.extra_data} == {"A", "B"}


@pytest.mark.parametrize("scenario_class", SCENARIOS)
def test_each_scenario_metadata_uses_its_own_name(scenario_class: Type[PR26BaseScenario]) -> None:
    metadata = scenario_class(data_path="unused.xlsx").get_metadata()
    assert metadata.name == scenario_class.name
    assert metadata.display_name == scenario_class.display_name
    if scenario_class is not PR26ControlScenario:
        assert metadata.main_metric == "medrobustness_fully_correct_rate"


def test_control_ranks_on_noise_floor_and_others_on_fully_correct_rate() -> None:
    assert PR26ControlScenario(data_path="unused.xlsx").get_metadata().main_metric == (
        "medrobustness_control_failure_rate"
    )
    for scenario_class in SCENARIOS:
        if scenario_class is not PR26ControlScenario:
            metadata = scenario_class(data_path="unused.xlsx").get_metadata()
            assert metadata.main_metric == "medrobustness_fully_correct_rate"
