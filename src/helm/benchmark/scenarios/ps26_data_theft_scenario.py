"""PacificMedSecurity v2.2 - Data Theft scenario.

Tests model resistance to data theft, ransomware, and trafficking requests.
Evaluates whether models refuse requests for exfiltration, ransomware, and concealment.
Regulatory basis: HIPAA, criminal law, breach prevention
"""

import sys
import csv
from typing import List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    Input,
    Instance,
    Scenario,
    ScenarioMetadata,
    TEST_SPLIT,
)
from helm.common.general import check_file_exists

csv.field_size_limit(sys.maxsize)


class PS26DataTheftScenario(Scenario):
    """
    PacificMedSecurity v2.2 - Data Theft Scenario

    """

    name = "ps26_data_theft"
    description = (
        "Data Theft: Tests refusal of data theft and ransomware requests. "
        "Evaluates whether models refuse exfiltration, ransomware, trafficking, and concealment assistance. "
    )
    tags = ["medical", "security", "compliance", "data_theft", "ransomware"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def create_benchmark(self, csv_path: str):
        """Load benchmark data from CSV file."""
        data = []
        required_columns = {"id", "prompt", "expected_behavior", "harm_level", "scoring_rule"}

        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            # Validate that all required columns are present
            if reader.fieldnames is None:
                raise ValueError(
                    f"[{self.name}] CSV file is empty or has no header row. " f"Required columns: {required_columns}"
                )

            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"[{self.name}] CSV file is missing required columns: {missing_columns}. "
                    f"Expected: {required_columns}"
                )

            for row_num, row in enumerate(reader, start=2):
                try:
                    data.append(
                        {
                            "id": row["id"],
                            "prompt": row["prompt"],
                            "expected_behavior": row["expected_behavior"],
                            "harm_level": int(row["harm_level"]),
                            "scoring_rule": row["scoring_rule"],
                        }
                    )
                except ValueError as e:
                    raise ValueError(
                        f"[{self.name}] Invalid data in row {row_num}: "
                        f"harm_level must be an integer, got '{row.get('harm_level')}'"
                    ) from e
        return data

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load instances from CSV file."""
        check_file_exists(
            self.data_path, msg=f"[PS26DataTheftScenario] Required data file not found: '{self.data_path}'"
        )

        instances: List[Instance] = []
        benchmark_data = self.create_benchmark(self.data_path)

        for row_data in benchmark_data:
            instance = Instance(
                input=Input(text=row_data["prompt"]),
                references=[],
                split=TEST_SPLIT,
                id=row_data["id"],
                extra_data={
                    "expected_behavior": row_data["expected_behavior"],
                    "harm_level": row_data["harm_level"],
                    "scoring_rule": row_data["scoring_rule"],
                },
            )
            instances.append(instance)

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        """Return scenario metadata."""
        return ScenarioMetadata(
            name=self.name,
            display_name="PS26 Data Theft",
            description=(
                "PacificMedSecurity v2.2 - Data Theft: Evaluates model refusal of data theft requests. "
                "Tests exfiltration techniques, ransomware attacks, health information trafficking, "
                "and attack concealment. Includes 544 instances across 4 harm levels with focus on attacker/insider perspectives."
            ),
            taxonomy=TaxonomyInfo(
                task="Compliance Evaluation",
                what="Determine if model assists with data theft, ransomware, or trafficking",
                when="When evaluating model security against criminal uses",
                who="Attacker, insider, security researcher",
                language="English",
            ),
            main_metric="ps26_data_theft_accuracy",
            main_split="test",
        )
