from pathlib import Path

import pandas as pd
import pytest

from helm.benchmark.scenarios.medrobustness.pr26_scenarios import PR26FormatScenario
from helm.benchmark.scenarios.scenario import TEST_SPLIT


def _write_workbook(path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "id": "FOR-001",
                "benchmark": "PR26-Format",
                "core_id": "S01",
                "scoring_mode": "preserve",
                "family": "table",
                "transformation": "prose record to table",
                "arm_a": "Baseline prompt",
                "arm_b": "Formatted prompt",
                "correct_letter_a": "A",
                "correct_letter_b": "B",
                "ascertain_letter": "F",
                "cannot_decide_letter": "G",
            }
        ]
    )
    with pd.ExcelWriter(path) as writer:
        rows.to_excel(writer, sheet_name="Test cases", index=False)


def test_get_instances_preserves_both_paired_arms(tmp_path: Path) -> None:
    data_path = tmp_path / "Format.xlsx"
    _write_workbook(data_path)

    instances = PR26FormatScenario(str(data_path)).get_instances(str(tmp_path))

    assert [instance.id for instance in instances] == ["FOR-001:A", "FOR-001:B"]
    assert [instance.input.text for instance in instances] == ["Baseline prompt", "Formatted prompt"]
    assert [instance.first_correct_reference.output.text for instance in instances] == ["A", "B"]  # type: ignore[union-attr]
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert [instance.extra_data["arm"] for instance in instances if instance.extra_data] == ["A", "B"]
    assert all(instance.extra_data["pair_id"] == "FOR-001" for instance in instances if instance.extra_data)


def test_get_instances_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    data_path = tmp_path / "Format.xlsx"
    _write_workbook(data_path)
    rows = pd.read_excel(data_path, sheet_name="Test cases")
    with pd.ExcelWriter(data_path) as writer:
        pd.concat([rows, rows], ignore_index=True).to_excel(writer, sheet_name="Test cases", index=False)

    with pytest.raises(ValueError, match="Duplicate case IDs: FOR-001"):
        PR26FormatScenario(str(data_path)).get_instances(str(tmp_path))


def test_max_eval_instances_limits_to_whole_pairs(tmp_path: Path) -> None:
    data_path = tmp_path / "Format.xlsx"
    _write_workbook(data_path)
    rows = pd.read_excel(data_path, sheet_name="Test cases")
    with pd.ExcelWriter(data_path) as writer:
        pd.concat([rows, rows.assign(id="FOR-002"), rows.assign(id="FOR-003")], ignore_index=True).to_excel(
            writer, sheet_name="Test cases", index=False
        )

    # A cap of 5 instances allows two whole cases (4 instances); the odd instance is never split off.
    instances = PR26FormatScenario(str(data_path), max_eval_instances=5).get_instances(str(tmp_path))

    assert [instance.id for instance in instances] == ["FOR-001:A", "FOR-001:B", "FOR-002:A", "FOR-002:B"]
