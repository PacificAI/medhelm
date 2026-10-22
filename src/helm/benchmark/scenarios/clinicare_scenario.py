"""CliniCARE-Bench scenario: one MedHELM instance per built CliniCARE task."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.clinicare_constants import (
    CLINICARE_ROOT_ENV,
    CLINICARE_TASKS_DIR_ENV,
    VERDICT_LABELS,
)
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
    ScenarioMetadata,
)

# Task ids are "<scenario_id>-<6 hex>". The hash covers the case parameters and is computed by
# CliniCARE's build_tasks.py; it is read from directory names here, never recomputed.
TASK_ID_PATTERN = re.compile(r"^(?P<scenario_id>[a-z0-9]+(?:-[a-z0-9]+)*)-[0-9a-f]{6}$")


def resolve_clinicare_root(explicit: str = "") -> Optional[Path]:
    """The CliniCARE checkout: ``clinicare_root`` argument, then ``CLINICARE_ROOT``; None if unset."""
    raw = (explicit or "").strip() or (os.environ.get(CLINICARE_ROOT_ENV) or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def resolve_tasks_dir(explicit: str = "", clinicare_root: str = "") -> Path:
    """Locate a built CliniCARE tasks dir (``benchmark/tasks/`` in a CliniCARE checkout).

    Precedence: ``tasks_dir`` argument, ``CLINICARE_TASKS_DIR``, then ``<CliniCARE root>/benchmark/tasks``.
    """
    raw = (explicit or "").strip() or (os.environ.get(CLINICARE_TASKS_DIR_ENV) or "").strip()
    root = resolve_clinicare_root(clinicare_root)
    if not raw and root is not None:
        raw = str(root / "benchmark" / "tasks")
    if not raw:
        raise FileNotFoundError(
            "No CliniCARE tasks dir given. Set tasks_dir= or clinicare_root= on the run entry, or export "
            f"{CLINICARE_TASKS_DIR_ENV} or {CLINICARE_ROOT_ENV}. Build tasks with CliniCARE's "
            "benchmark/scripts/build_tasks.py."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"CliniCARE tasks dir not found: {path}")
    return path


def parse_scenario_id(task_id: str) -> str:
    """``advanced-imaging-000001`` -> ``advanced-imaging``."""
    match = TASK_ID_PATTERN.match(task_id)
    if not match:
        raise ValueError(f"Not a CliniCARE task id (expected <scenario_id>-<6 hex>): {task_id!r}")
    return match.group("scenario_id")


def parse_task_ids(task_ids: str) -> List[str]:
    """Parse a '+'-separated task id list from a run entry (commas are reserved)."""
    if not task_ids or not str(task_ids).strip():
        return []
    return [part.strip() for part in str(task_ids).replace(",", "+").split("+") if part.strip()]


def load_cohort_labels(cohort_csv: str) -> Dict[str, Dict[str, str]]:
    """Read gold verdicts from the cohort bundle's ``full750.csv``, keyed by task id.

    ``full750.csv`` is the label source of truth for every build, including dev148: the bundle's
    ``dev148.csv`` lists subset membership only and carries no labels.
    """
    path = Path(cohort_csv).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CliniCARE cohort CSV not found: {path}")
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"task_id", "scenario_id", "expected_label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} has no {sorted(missing)} column(s); cohort_csv must be the bundle's full750.csv")
        for row in reader:
            label = row["expected_label"].strip()
            if label not in VERDICT_LABELS:
                raise ValueError(f"Unknown expected_label {label!r} in {path}")
            rows[row["task_id"].strip()] = {"scenario_id": row["scenario_id"].strip(), "label": label}
    return rows


class CliniCAREScenario(Scenario):
    """Wrap the task dirs of a CliniCARE build as MedHELM instances.

    The tasks dir is the coverage denominator: every task in it becomes an instance, whether or not
    the agent run produced a trial for it. Enumerate the tasks dir, never a job dir.
    """

    name = "clinicare"
    description = (
        "CliniCARE-Bench evaluates agents that investigate one real MIMIC-IV patient record through a "
        "governed tool surface and return a four-way verdict (Yes / No / Indeterminate - lack of data / "
        "Indeterminate - medical ambiguity) with cited evidence."
    )
    tags = ["agentic", "biomedical", "ehr"]

    def __init__(self, tasks_dir: str = "", cohort_csv: str = "", task_ids: str = "", clinicare_root: str = ""):
        super().__init__()
        self.tasks_dir = tasks_dir
        self.clinicare_root = clinicare_root
        self.cohort_csv = cohort_csv
        self.task_ids = parse_task_ids(task_ids)

    def get_instances(self, output_path: str) -> List[Instance]:
        del output_path  # tasks live in the CliniCARE build, not HELM's scenario cache
        tasks_dir = resolve_tasks_dir(self.tasks_dir, self.clinicare_root)
        cohort: Optional[Dict[str, Dict[str, str]]] = load_cohort_labels(self.cohort_csv) if self.cohort_csv else None
        allowed_ids = set(self.task_ids)
        instances: List[Instance] = []

        for task_dir in sorted(tasks_dir.iterdir()):
            # Skips _sidecar.compose.yml and any other non-task entries at the top level.
            if not task_dir.is_dir() or task_dir.name.startswith(("_", ".")):
                continue
            task_id = task_dir.name
            if allowed_ids and task_id not in allowed_ids:
                continue
            scenario_id = parse_scenario_id(task_id)
            instruction = task_dir / "instruction.md"
            if not instruction.is_file():
                raise FileNotFoundError(f"CliniCARE task {task_id} has no instruction.md; rebuild the tasks dir")

            references: List[Reference] = []
            if cohort is not None:
                row = cohort.get(task_id)
                if row is None:
                    raise ValueError(
                        f"Task {task_id} is in {tasks_dir} but not in {self.cohort_csv}; "
                        "the tasks dir and cohort CSV are from different builds"
                    )
                if row["scenario_id"] != scenario_id:
                    raise ValueError(
                        f"Cohort CSV gives scenario {row['scenario_id']!r} for task {task_id}, "
                        f"expected {scenario_id!r}"
                    )
                references = [Reference(Output(text=row["label"]), tags=[CORRECT_TAG])]

            instances.append(
                Instance(
                    # Task ids identify the patient to anyone holding the CliniCARE repo: never
                    # publish per-instance outputs (see LEADERBOARD_EXPORT.md).
                    id=task_id,
                    # The exact prompt the agent saw, read verbatim rather than re-rendered.
                    input=Input(text=instruction.read_text(encoding="utf-8")),
                    references=references,
                    split=TEST_SPLIT,
                    extra_data={"task_id": task_id, "scenario_id": scenario_id},
                )
            )

        if not instances:
            raise ValueError(f"No CliniCARE tasks matched task_ids={self.task_ids!r} under {tasks_dir}")
        if allowed_ids:
            missing = allowed_ids - {instance.id for instance in instances}
            if missing:
                raise ValueError(f"Requested task ids not found under {tasks_dir}: {sorted(missing)}")
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="clinicare",
            display_name="CliniCARE-Bench",
            description=(
                "CliniCARE-Bench evaluates agents on 750 clinical questions (25 scenarios x 30 patients) "
                "over real MIMIC-IV records. The agent reaches the chart through a governed tool surface "
                "and must return a four-way verdict with cited evidence; 150 cases admit no defensible "
                "Yes or No."
            ),
            taxonomy=TaxonomyInfo(
                task="Agentic clinical decision support",
                what="Investigate one patient's EHR with tools and adjudicate a clinical question",
                when="Any",
                who="Clinician",
                language="English",
            ),
            main_metric="clinicare_score",
            main_split="test",
        )
