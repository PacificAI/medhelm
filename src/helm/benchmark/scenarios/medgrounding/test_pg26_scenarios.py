from pathlib import Path
from typing import List, Type

import pandas as pd
import pytest

from helm.benchmark.scenarios.medgrounding.pg26_base_scenarios import PG26BaseScenario
from helm.benchmark.scenarios.medgrounding.pg26_scenarios import PG26AblationScenario
from helm.benchmark.scenarios.medgrounding.pg26_scenarios import PG26AbstentionScenario
from helm.benchmark.scenarios.medgrounding.pg26_scenarios import PG26ContradictionScenario
from helm.benchmark.scenarios.medgrounding.pg26_scenarios import PG26InversionScenario
from helm.benchmark.scenarios.medgrounding.pg26_scenarios import PG26MisfiledScenario
from helm.benchmark.scenarios.medgrounding.pg26_scenarios import PG26StalenessScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT

SINGLE_SCENARIOS: List[Type[PG26BaseScenario]] = [
    PG26AblationScenario,
    PG26AbstentionScenario,
    PG26ContradictionScenario,
    PG26MisfiledScenario,
    PG26StalenessScenario,
]


def _write_single_workbook(path: Path, case_ids: List[str], correct_letter: str = "A") -> None:
    rows = pd.DataFrame(
        [
            {
                "id": case_id,
                "benchmark": "PG26-Test",
                "core_id": "S01",
                "record": f"Record {case_id}",
                "correct_letter": correct_letter,
                "ascertain_letter": "F",
                "cannot_decide_letter": "G",
            }
            for case_id in case_ids
        ]
    )
    with pd.ExcelWriter(path) as writer:
        rows.to_excel(writer, sheet_name="Records", index=False)


def _write_paired_workbook(path: Path, case_ids: List[str], correct_letter_b: str = "B") -> None:
    rows = pd.DataFrame(
        [
            {
                "id": case_id,
                "benchmark": "PG26-Inversion",
                "core_id": "S01",
                "arm_a": f"Full {case_id}",
                "arm_b": f"Inverted {case_id}",
                "correct_letter_a": "A",
                "correct_letter_b": correct_letter_b,
                "ascertain_letter": "F",
                "cannot_decide_letter": "G",
            }
            for case_id in case_ids
        ]
    )
    with pd.ExcelWriter(path) as writer:
        rows.to_excel(writer, sheet_name="Test cases", index=False)


@pytest.mark.parametrize("scenario_class", SINGLE_SCENARIOS)
def test_single_scenarios_emit_one_instance_per_case_and_respect_cap(
    tmp_path: Path, scenario_class: Type[PG26BaseScenario]
) -> None:
    data_path = tmp_path / "workbook.xlsx"
    _write_single_workbook(data_path, ["TST-001", "TST-002", "TST-003"])

    instances = scenario_class(str(data_path)).get_instances("unused")
    assert [instance.id for instance in instances] == ["TST-001", "TST-002", "TST-003"]
    assert {instance.split for instance in instances} == {TEST_SPLIT}
    assert [instance.input.text for instance in instances] == ["Record TST-001", "Record TST-002", "Record TST-003"]

    capped = scenario_class(str(data_path), max_eval_instances=2).get_instances("unused")
    assert [instance.id for instance in capped] == ["TST-001", "TST-002"]


def test_inversion_emits_whole_pairs_and_respects_cap(tmp_path: Path) -> None:
    data_path = tmp_path / "Inversion.xlsx"
    _write_paired_workbook(data_path, ["INV-001", "INV-002"])

    instances = PG26InversionScenario(str(data_path)).get_instances("unused")
    assert [instance.id for instance in instances] == ["INV-001:A", "INV-001:B", "INV-002:A", "INV-002:B"]
    assert [instance.first_correct_reference.output.text for instance in instances] == [  # type: ignore[union-attr]
        "A",
        "B",
        "A",
        "B",
    ]

    capped = PG26InversionScenario(str(data_path), max_eval_instances=3).get_instances("unused")
    assert [instance.id for instance in capped] == ["INV-001:A", "INV-001:B"]


def test_scenario_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "Ablation.xlsx"
    _write_single_workbook(data_path, ["ABL-001", "ABL-001"])

    with pytest.raises(ValueError, match="Duplicate case IDs: ABL-001"):
        PG26AblationScenario(str(data_path)).get_instances("unused")


def test_scenario_rejects_invalid_letter(tmp_path: Path) -> None:
    data_path = tmp_path / "Abstention.xlsx"
    _write_single_workbook(data_path, ["ABS-001"], correct_letter="Z")

    with pytest.raises(ValueError, match="invalid letter 'Z' in 'correct_letter'"):
        PG26AbstentionScenario(str(data_path)).get_instances("unused")


def test_every_scenario_uses_the_shared_main_metric() -> None:
    for scenario_class in SINGLE_SCENARIOS + [PG26InversionScenario]:
        metadata = scenario_class(data_path="unused.xlsx").get_metadata()
        assert metadata.name == scenario_class.name
        assert metadata.main_metric == "medgrounding_fully_correct_rate"
