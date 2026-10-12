from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
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
from helm.common.general import check_file_exists


class PR26FormatScenario(Scenario):
    """PacificMedRobustness PR26 Format paired benchmark."""

    name = "pr26_format"
    description = (
        "Evaluates whether clinically equivalent prose and tabular records produce the same correct management "
        "decision."
    )
    tags = ["knowledge", "reasoning", "biomedical", "multi-hop", "robustness"]

    ARMS_PER_CASE = 2

    _REQUIRED_COLUMNS = {
        "id",
        "benchmark",
        "core_id",
        "scoring_mode",
        "family",
        "transformation",
        "arm_a",
        "arm_b",
        "correct_letter_a",
        "correct_letter_b",
        "ascertain_letter",
        "cannot_decide_letter",
    }

    def __init__(
        self,
        data_path: str,
        max_cases: Optional[int] = None,
        max_eval_instances: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to the PR26 workbook.
            max_cases: If set, only the first `max_cases` case rows are used (for quick test runs). Each case
                contributes both of its arms, so the paired metric always sees complete pairs.
            max_eval_instances: Instance cap from `helm-run --max-eval-instances`. Each case yields two instances,
                so this is converted to a whole number of cases; the runner never needs to sample.
        """
        super().__init__()
        if max_cases is not None and max_cases < 1:
            raise ValueError(f"[PR26FormatScenario] max_cases must be at least 1, got {max_cases}")
        if max_eval_instances is not None:
            cases_from_cap = max_eval_instances // self.ARMS_PER_CASE
            if cases_from_cap < 1:
                raise ValueError(
                    f"[PR26FormatScenario] max_eval_instances must be at least {self.ARMS_PER_CASE}, "
                    f"got {max_eval_instances}"
                )
            max_cases = cases_from_cap if max_cases is None else min(max_cases, cases_from_cap)
        self.data_path = data_path
        self.max_cases = max_cases

    def _load_test_cases(self) -> pd.DataFrame:
        check_file_exists(
            self.data_path,
            msg=f"[PR26FormatScenario] Required data file not found: '{self.data_path}'",
        )
        rows = pd.read_excel(self.data_path, sheet_name="Test cases")
        missing_columns = sorted(self._REQUIRED_COLUMNS - set(rows.columns))
        if missing_columns:
            raise ValueError(
                f"[PR26FormatScenario] Missing required columns in '{self.data_path}': " f"{', '.join(missing_columns)}"
            )
        if rows["id"].duplicated().any():
            duplicate_ids = sorted(rows.loc[rows["id"].duplicated(keep=False), "id"].astype(str).unique())
            raise ValueError(f"[PR26FormatScenario] Duplicate case IDs: {', '.join(duplicate_ids)}")
        return rows

    @staticmethod
    def _required_text(row: pd.Series, column: str) -> str:
        value = row[column]
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"[PR26FormatScenario] Case '{row['id']}' has no value for '{column}'")
        return str(value).strip()

    def get_instances(self, output_path: str) -> List[Instance]:
        del output_path
        instances: List[Instance] = []

        cases = self._load_test_cases()
        if self.max_cases is not None:
            cases = cases.head(self.max_cases)

        for _, row in cases.iterrows():
            case_id = self._required_text(row, "id")
            common_extra_data: Dict[str, Any] = {
                "pair_id": case_id,
                "benchmark": self._required_text(row, "benchmark"),
                "core_id": self._required_text(row, "core_id"),
                "scoring_mode": self._required_text(row, "scoring_mode"),
                "family": self._required_text(row, "family"),
                "transformation": self._required_text(row, "transformation"),
                "ascertain_letter": self._required_text(row, "ascertain_letter").upper(),
                "cannot_decide_letter": self._required_text(row, "cannot_decide_letter").upper(),
            }
            for arm in ("a", "b"):
                correct_letter = self._required_text(row, f"correct_letter_{arm}").upper()
                if correct_letter not in "ABCDEFG":
                    raise ValueError(
                        f"[PR26FormatScenario] Case '{case_id}' arm {arm.upper()} has invalid correct letter "
                        f"'{correct_letter}'"
                    )
                instances.append(
                    Instance(
                        id=f"{case_id}:{arm.upper()}",
                        input=Input(text=self._required_text(row, f"arm_{arm}")),
                        references=[
                            Reference(output=Output(text=correct_letter), tags=[CORRECT_TAG]),
                        ],
                        split=TEST_SPLIT,
                        extra_data={**common_extra_data, "arm": arm.upper()},
                    )
                )

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name=self.name,
            display_name="PacificMedRobustness: Format",
            description=self.description,
            taxonomy=TaxonomyInfo(
                task="Multiple Choice Question Answering",
                what="Assess robustness of clinical management decisions to record formatting",
                when="Real-time clinical decision making",
                who="Clinician",
                language="English",
            ),
            main_metric="medrobustness_fully_correct_rate",
            main_split=TEST_SPLIT,
        )

    def __repr__(self) -> str:
        return f"PR26FormatScenario(data_path={Path(self.data_path)!s}, max_cases={self.max_cases})"
