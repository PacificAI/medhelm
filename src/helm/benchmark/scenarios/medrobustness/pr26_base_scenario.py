from pathlib import Path
from typing import Dict, List, Optional

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

ARMS = ("A", "B")
VALID_LETTERS = frozenset("ABCDEFG")
REQUIRED_COLUMNS = frozenset(
    {
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
)
CASE_TEXT_COLUMNS = ("benchmark", "core_id", "scoring_mode", "family", "transformation")
CASE_LETTER_COLUMNS = ("ascertain_letter", "cannot_decide_letter")


class PR26BaseScenario(Scenario):
    """Shared loader for PacificMedRobustness PR26 paired benchmarks.

    Each case yields two instances (arms A and B) with the same `pair_id`, so the paired metric always sees
    complete pairs. Subclasses set the class attributes below and nothing else.
    """

    display_name: str
    what: str
    main_metric: str = "medrobustness_fully_correct_rate"
    tags = ["knowledge", "reasoning", "biomedical", "multi-hop", "robustness"]

    def __init__(self, data_path: str, max_eval_instances: Optional[int] = None):
        """
        Args:
            data_path: Path to the PR26 workbook.
            max_eval_instances: If set, use only the cases that fit in this many instances. Each case yields
                two instances, so the cap is rounded down to whole cases.
        """
        super().__init__()
        self.data_path = data_path
        self.max_eval_instances = max_eval_instances

    def _load_test_cases(self) -> pd.DataFrame:
        check_file_exists(
            self.data_path,
            msg=f"[{type(self).__name__}] Required data file not found: '{self.data_path}'",
        )
        rows = pd.read_excel(self.data_path, sheet_name="Test cases")
        missing_columns = sorted(REQUIRED_COLUMNS - set(rows.columns))
        if missing_columns:
            raise ValueError(
                f"[{type(self).__name__}] Missing required columns in '{self.data_path}': "
                f"{', '.join(missing_columns)}"
            )
        duplicated = rows["id"].duplicated(keep=False)
        if duplicated.any():
            duplicate_ids = sorted(rows.loc[duplicated, "id"].astype(str).unique())
            raise ValueError(f"[{type(self).__name__}] Duplicate case IDs: {', '.join(duplicate_ids)}")
        return rows

    @staticmethod
    def _required_text(row: pd.Series, column: str) -> str:
        value = row[column]
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"Case '{row['id']}' has no value for '{column}'")
        return str(value).strip()

    def _build_case_instances(self, row: pd.Series) -> List[Instance]:
        case_id = self._required_text(row, "id")
        case_data: Dict[str, str] = {"pair_id": case_id}
        case_data.update({column: self._required_text(row, column) for column in CASE_TEXT_COLUMNS})
        case_data.update({column: self._required_text(row, column).upper() for column in CASE_LETTER_COLUMNS})

        instances: List[Instance] = []
        for arm in ARMS:
            correct_letter = self._required_text(row, f"correct_letter_{arm.lower()}").upper()
            if correct_letter not in VALID_LETTERS:
                raise ValueError(f"Case '{case_id}' arm {arm} has invalid correct letter '{correct_letter}'")
            instances.append(
                Instance(
                    id=f"{case_id}:{arm}",
                    input=Input(text=self._required_text(row, f"arm_{arm.lower()}")),
                    references=[Reference(output=Output(text=correct_letter), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                    extra_data={**case_data, "arm": arm},
                )
            )
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        del output_path
        cases = self._load_test_cases()
        if self.max_eval_instances is not None:
            cases = cases.head(self.max_eval_instances // len(ARMS))
        return [instance for _, row in cases.iterrows() for instance in self._build_case_instances(row)]

    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            taxonomy=TaxonomyInfo(
                task="Multiple Choice Question Answering",
                what=self.what,
                when="Real-time clinical decision making",
                who="Clinician",
                language="English",
            ),
            main_metric=self.main_metric,
            main_split=TEST_SPLIT,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(data_path={Path(self.data_path)!s}, "
            f"max_eval_instances={self.max_eval_instances})"
        )
