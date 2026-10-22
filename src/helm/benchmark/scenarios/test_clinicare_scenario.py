import csv
from pathlib import Path
from typing import Dict, Tuple

import pytest

from helm.benchmark.scenarios.clinicare_constants import CLINICARE_TASKS_DIR_ENV
from helm.benchmark.scenarios.clinicare_scenario import (
    CliniCAREScenario,
    parse_scenario_id,
    resolve_tasks_dir,
)
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT

# Placeholder tasks: the real instruction.md quotes a patient record and must never be a fixture.
TASKS: Dict[str, str] = {
    "advanced-imaging-000001": "Placeholder instruction A.\n\nLine two.\n",
    "advanced-imaging-000002": "Placeholder instruction B.\n",
    "sepsis-bundle-000003": "Placeholder instruction C.\n",
}

# task_id -> (scenario_id, expected_label)
COHORT_ROWS: Dict[str, Tuple[str, str]] = {
    "advanced-imaging-000001": ("advanced-imaging", "YES"),
    "advanced-imaging-000002": ("advanced-imaging", "INDETERMINATE_LACK_OF_DATA"),
    "sepsis-bundle-000003": ("sepsis-bundle", "NO"),
}


def _write_tasks_dir(root: Path) -> Path:
    tasks_dir = root / "tasks"
    for task_id, instruction in TASKS.items():
        task_dir = tasks_dir / task_id
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
        (task_dir / "task.toml").write_text(f'[task]\nname = "mimic/{task_id}"\n', encoding="utf-8")
    (tasks_dir / "_sidecar.compose.yml").write_text("services: {}\n", encoding="utf-8")
    (tasks_dir / "notes.txt").write_text("stray file\n", encoding="utf-8")
    return tasks_dir


def _write_cohort(root: Path, rows: Dict[str, Tuple[str, str]]) -> Path:
    path = root / "full750.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "scenario_id", "params_json", "expected_label"])
        for task_id, (scenario_id, label) in rows.items():
            writer.writerow([task_id, scenario_id, "{}", label])
    return path


def test_one_instance_per_task_dir(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    instances = CliniCAREScenario(tasks_dir=str(tasks_dir)).get_instances(str(tmp_path / "out"))

    assert [instance.id for instance in instances] == sorted(TASKS)
    for instance in instances:
        assert instance.id is not None and instance.extra_data is not None
        assert instance.split == TEST_SPLIT
        assert instance.input.text == TASKS[instance.id]
        assert instance.references == []
        assert instance.extra_data["task_id"] == instance.id


def test_scenario_id_keeps_hyphens():
    assert parse_scenario_id("advanced-imaging-000001") == "advanced-imaging"
    assert parse_scenario_id("stroke-time-targets-0a1b2c") == "stroke-time-targets"
    assert parse_scenario_id("readmission-30d-abcdef") == "readmission-30d"


@pytest.mark.parametrize("bad_id", ["advanced-imaging", "advanced-imaging-0091", "advanced-imaging-0091fg", "_sidecar"])
def test_scenario_id_rejects_non_task_ids(bad_id):
    with pytest.raises(ValueError):
        parse_scenario_id(bad_id)


def test_extra_data_scenario_id(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    instances = CliniCAREScenario(tasks_dir=str(tasks_dir)).get_instances(str(tmp_path / "out"))
    scenario_ids = {instance.id: (instance.extra_data or {}).get("scenario_id") for instance in instances}

    assert scenario_ids["advanced-imaging-000001"] == "advanced-imaging"
    assert scenario_ids["sepsis-bundle-000003"] == "sepsis-bundle"


def test_tasks_dir_from_env(tmp_path, monkeypatch):
    tasks_dir = _write_tasks_dir(tmp_path)
    monkeypatch.setenv(CLINICARE_TASKS_DIR_ENV, str(tasks_dir))

    assert resolve_tasks_dir() == tasks_dir.resolve()
    assert len(CliniCAREScenario().get_instances(str(tmp_path / "out"))) == len(TASKS)


def test_missing_tasks_dir_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(CLINICARE_TASKS_DIR_ENV, raising=False)
    with pytest.raises(FileNotFoundError):
        resolve_tasks_dir()
    with pytest.raises(FileNotFoundError):
        resolve_tasks_dir(str(tmp_path / "does-not-exist"))


def test_task_without_instruction_raises(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    (tasks_dir / "advanced-imaging-000002" / "instruction.md").unlink()
    with pytest.raises(FileNotFoundError):
        CliniCAREScenario(tasks_dir=str(tasks_dir)).get_instances(str(tmp_path / "out"))


def test_task_ids_filter(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    scenario = CliniCAREScenario(tasks_dir=str(tasks_dir), task_ids="sepsis-bundle-000003")
    assert [instance.id for instance in scenario.get_instances(str(tmp_path / "out"))] == ["sepsis-bundle-000003"]


def test_task_ids_filter_unknown_id_raises(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    scenario = CliniCAREScenario(tasks_dir=str(tasks_dir), task_ids="sepsis-bundle-000003+sepsis-bundle-ffffff")
    with pytest.raises(ValueError):
        scenario.get_instances(str(tmp_path / "out"))


def test_cohort_supplies_gold_verdicts(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    cohort = _write_cohort(tmp_path, COHORT_ROWS)
    instances = CliniCAREScenario(tasks_dir=str(tasks_dir), cohort_csv=str(cohort)).get_instances(str(tmp_path / "out"))
    gold = {instance.id: instance.references for instance in instances}

    assert [ref.output.text for ref in gold["advanced-imaging-000001"]] == ["YES"]
    assert [ref.output.text for ref in gold["advanced-imaging-000002"]] == ["INDETERMINATE_LACK_OF_DATA"]
    assert all(CORRECT_TAG in refs[0].tags for refs in gold.values())


def test_cohort_rows_without_tasks_are_allowed(tmp_path):
    # A dev148 tasks dir scored against the full750 cohort: extra cohort rows are fine.
    tasks_dir = _write_tasks_dir(tmp_path)
    cohort = _write_cohort(tmp_path, {**COHORT_ROWS, "pe-workup-123456": ("pe-workup", "NO")})
    instances = CliniCAREScenario(tasks_dir=str(tasks_dir), cohort_csv=str(cohort)).get_instances(str(tmp_path / "out"))
    assert len(instances) == len(TASKS)


def test_task_missing_from_cohort_raises(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    rows = dict(COHORT_ROWS)
    del rows["sepsis-bundle-000003"]
    cohort = _write_cohort(tmp_path, rows)
    with pytest.raises(ValueError, match="different builds"):
        CliniCAREScenario(tasks_dir=str(tasks_dir), cohort_csv=str(cohort)).get_instances(str(tmp_path / "out"))


def test_cohort_scenario_mismatch_raises(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    cohort = _write_cohort(tmp_path, {**COHORT_ROWS, "sepsis-bundle-000003": ("pe-workup", "NO")})
    with pytest.raises(ValueError):
        CliniCAREScenario(tasks_dir=str(tasks_dir), cohort_csv=str(cohort)).get_instances(str(tmp_path / "out"))


def test_cohort_unknown_label_raises(tmp_path):
    tasks_dir = _write_tasks_dir(tmp_path)
    cohort = _write_cohort(tmp_path, {**COHORT_ROWS, "sepsis-bundle-000003": ("sepsis-bundle", "MAYBE")})
    with pytest.raises(ValueError):
        CliniCAREScenario(tasks_dir=str(tasks_dir), cohort_csv=str(cohort)).get_instances(str(tmp_path / "out"))


def test_membership_only_csv_raises(tmp_path):
    # dev148.csv lists task ids only; labels always come from full750.csv.
    tasks_dir = _write_tasks_dir(tmp_path)
    dev = tmp_path / "dev148.csv"
    dev.write_text("task_id\n" + "".join(f"{task_id}\n" for task_id in TASKS), encoding="utf-8")
    with pytest.raises(ValueError, match="full750.csv"):
        CliniCAREScenario(tasks_dir=str(tasks_dir), cohort_csv=str(dev)).get_instances(str(tmp_path / "out"))


def test_metadata():
    metadata = CliniCAREScenario().get_metadata()
    assert metadata.name == "clinicare"
    assert metadata.main_metric == "clinicare_score"
    assert metadata.main_split == "test"
