from typing import ClassVar, Dict, List, Optional

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

VALID_LETTERS = frozenset("ABCDEFG")
ARMS = ("A", "B")
COMMON_COLUMNS = frozenset({"id", "benchmark", "core_id", "ascertain_letter", "cannot_decide_letter"})
SINGLE_COLUMNS = frozenset({"record", "correct_letter"})
PAIRED_COLUMNS = frozenset({"arm_a", "arm_b", "correct_letter_a", "correct_letter_b"})
OPTIONAL_CASE_COLUMNS = ("correct_letter_full_record", "key_policy", "case_kind", "hop_removed", "hop_inverted")
CASE_LETTER_COLUMNS = ("ascertain_letter", "cannot_decide_letter")


class PG26BaseScenario(Scenario):
    """Shared loader for PacificMedGrounding PG26 benchmarks.

    Two layouts, selected by the `layout` class attribute:
      - "single": one record per case, scored against `correct_letter` (sheet `Records`).
      - "paired": two arms per case (`arm_a`, `arm_b`), each scored against its own letter (sheet `Test cases`).
        Each case yields two instances sharing a `pair_id`.

    Subclasses set `display_name`, `what`, `sheet_name`, and `layout`, and nothing else.
    """

    display_name: str
    what: str
    sheet_name: ClassVar[str] = "Records"
    layout: ClassVar[str] = "single"
    main_metric: str = "medgrounding_fully_correct_rate"
    tags = ["knowledge", "reasoning", "biomedical", "multi-hop", "grounding"]

    def __init__(self, data_path: str, max_eval_instances: Optional[int] = None):
        """
        Args:
            data_path: Path to the PG26 workbook.
            max_eval_instances: If set, use only the cases that fit in this many instances. Paired cases yield
                two instances, so the cap is rounded down to whole cases.
        """
        super().__init__()
        self.data_path = data_path
        self.max_eval_instances = max_eval_instances

    def _load_cases(self) -> pd.DataFrame:
        check_file_exists(
            self.data_path,
            msg=f"[{type(self).__name__}] Required data file not found: '{self.data_path}'",
        )
        layout_columns = SINGLE_COLUMNS if self.layout == "single" else PAIRED_COLUMNS
        rows = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
        missing_columns = sorted((COMMON_COLUMNS | layout_columns) - set(rows.columns))
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

    @staticmethod
    def _valid_letter(case_id: str, column: str, letter: str) -> str:
        letter = letter.upper()
        if letter not in VALID_LETTERS:
            raise ValueError(f"Case '{case_id}' has invalid letter '{letter}' in '{column}'")
        return letter

    def _case_data(self, row: pd.Series) -> Dict[str, str]:
        case_id = self._required_text(row, "id")
        case_data: Dict[str, str] = {
            "pair_id": case_id,
            "benchmark": self._required_text(row, "benchmark"),
            "core_id": self._required_text(row, "core_id"),
        }
        for column in CASE_LETTER_COLUMNS:
            case_data[column] = self._valid_letter(case_id, column, self._required_text(row, column))
        for column in OPTIONAL_CASE_COLUMNS:
            if column in row.index and not pd.isna(row[column]):
                case_data[column] = str(row[column]).strip()
        return case_data

    def _build_single_instance(self, row: pd.Series) -> Instance:
        case_id = self._required_text(row, "id")
        correct_letter = self._valid_letter(case_id, "correct_letter", self._required_text(row, "correct_letter"))
        return Instance(
            id=case_id,
            input=Input(text=self._required_text(row, "record")),
            references=[Reference(output=Output(text=correct_letter), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
            extra_data=self._case_data(row),
        )

    def _build_paired_instances(self, row: pd.Series) -> List[Instance]:
        case_id = self._required_text(row, "id")
        case_data = self._case_data(row)
        instances: List[Instance] = []
        for arm in ARMS:
            column = f"correct_letter_{arm.lower()}"
            correct_letter = self._valid_letter(case_id, column, self._required_text(row, column))
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
        cases = self._load_cases()
        if self.max_eval_instances is not None:
            instances_per_case = 2 if self.layout == "paired" else 1
            cases = cases.head(self.max_eval_instances // instances_per_case)
        if self.layout == "paired":
            return [instance for _, row in cases.iterrows() for instance in self._build_paired_instances(row)]
        return [self._build_single_instance(row) for _, row in cases.iterrows()]

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
