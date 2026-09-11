"""PhysicianBench scenario: one MedHELM instance per EHR agent task."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.physician_bench_constants import (
    DEFAULT_FHIR_IMAGE,
    DEFAULT_PORT,
    DEFAULT_TASK_VERSION,
    PB_PROTOCOL,
    PB_ROOT_ENV,
)
from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Input,
    Instance,
    Scenario,
    ScenarioMetadata,
)


def resolve_pb_root(explicit: str = "") -> Path:
    """Locate the PhysicianBench checkout."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.environ.get(PB_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root).expanduser())
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "PhysicianBench",
            cwd.parent / "PhysicianBench",
        ]
    )
    # physician_bench_scenario.py -> scenarios -> benchmark -> helm -> src -> medhelm -> workspace
    here = Path(__file__).resolve()
    if len(here.parents) >= 6:
        candidates.append(here.parents[5] / "PhysicianBench")
    if len(here.parents) >= 5:
        candidates.append(here.parents[4].parent / "PhysicianBench")

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "tasks" / "v1").is_dir():
            return resolved

    raise FileNotFoundError(
        "Could not find PhysicianBench. Set PHYSICIAN_BENCH_ROOT or pass pb_root= "
        "to the physician_bench run spec."
    )


def parse_task_ids(task_ids: str) -> List[str]:
    """Parse a '+'-separated task id list from a run entry (commas are reserved)."""
    if not task_ids or not str(task_ids).strip():
        return []
    return [part.strip() for part in str(task_ids).replace(",", "+").split("+") if part.strip()]


def _goal_from_instruction(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return text.strip().split("\n", 1)[0][:200] if text.strip() else ""


class PhysicianBenchScenario(Scenario):
    """Wrap PhysicianBench v1 task folders as MedHELM instances.

    One instance = one task episode (Docker FHIR + MiniAgent + all pytest
    checkpoints for that task). ``--max-eval-instances`` caps how many tasks run.
    """

    name = "physician_bench"
    description = (
        "PhysicianBench evaluates LLM agents on physician tasks in a real EHR environment "
        "(FHIR APIs, long-horizon clinical workflows, pytest checkpoints)."
    )
    tags = ["health", "clinical", "agentic", "ehr"]

    def __init__(
        self,
        pb_root: str = "",
        version: str = DEFAULT_TASK_VERSION,
        task_ids: str = "",
        max_steps: str = "",
        fhir_image: str = "",
        port: str = "",
        reasoning_effort: str = "",
        temperature: str = "",
    ):
        super().__init__()
        self.pb_root = pb_root
        self.version = (version or DEFAULT_TASK_VERSION).strip()
        self.task_ids = parse_task_ids(task_ids)
        self.max_steps = str(max_steps).strip() if max_steps is not None else ""
        self.fhir_image = (fhir_image or "").strip()
        self.port = str(port).strip() if port is not None else ""
        self.reasoning_effort = (reasoning_effort or "").strip()
        self.temperature = str(temperature).strip() if temperature not in (None, "") else ""

    def _iter_task_dirs(self, pb_root: Path) -> List[Path]:
        tasks_root = pb_root / "tasks" / self.version
        if not tasks_root.is_dir():
            raise FileNotFoundError(f"PhysicianBench tasks not found: {tasks_root}")
        return sorted(path for path in tasks_root.iterdir() if path.is_dir() and (path / "instruction.md").is_file())

    def get_instances(self, output_path: str) -> List[Instance]:
        del output_path  # tasks live in the PhysicianBench checkout, not HELM's scenario cache
        pb_root = resolve_pb_root(self.pb_root)
        allowed_ids = set(self.task_ids)
        instances: List[Instance] = []

        for task_dir in self._iter_task_dirs(pb_root):
            task_id = task_dir.name
            if allowed_ids and task_id not in allowed_ids:
                continue
            instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
            relpath = task_dir.relative_to(pb_root).as_posix()
            envelope = {
                "pb_protocol": PB_PROTOCOL,
                "task_id": task_id,
                "task_relpath": relpath,
                "goal": _goal_from_instruction(instruction),
                "pb_root": str(pb_root),
                "fhir_image": self.fhir_image or DEFAULT_FHIR_IMAGE,
                "port": int(self.port) if self.port else DEFAULT_PORT,
            }
            if self.max_steps:
                envelope["max_steps"] = int(self.max_steps)
            if self.reasoning_effort:
                envelope["reasoning_effort"] = self.reasoning_effort
            if self.temperature:
                envelope["temperature"] = float(self.temperature)

            extra_data = {
                "task_relpath": relpath,
                "version": self.version,
            }
            instances.append(
                Instance(
                    id=task_id,
                    input=Input(text=json.dumps(envelope)),
                    references=[],
                    split=TEST_SPLIT,
                    extra_data=extra_data,
                )
            )

        if not instances:
            raise ValueError(
                f"No PhysicianBench tasks matched version={self.version!r}, "
                f"task_ids={self.task_ids!r} under {pb_root}"
            )
        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name="physician_bench",
            display_name="PhysicianBench",
            description=(
                "PhysicianBench evaluates LLM agents on 100 long-horizon physician tasks "
                "in a real EHR environment with FHIR APIs "
                "[(Liu et al., 2026)](https://arxiv.org/abs/2605.02240)."
            ),
            taxonomy=TaxonomyInfo(
                task="EHR agent evaluation",
                what="Complete physician workflows: retrieve EHR data, reason, order, document",
                when="Any",
                who="Clinician",
                language="English",
            ),
            main_metric="physician_bench_score",
            main_split="test",
        )
