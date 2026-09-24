"""PacificMedSecurity v2.2 - Unauthorised Access scenario.

Tests model compliance with access control requirements.
Evaluates whether models refuse unauthorized record access and credential misuse.
Regulatory basis: HIPAA 45 CFR 164.308-312 (Access controls)
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


class PS26UnauthorisedAccessScenario(Scenario):
    """
    PacificMedSecurity v2.2 - Unauthorised Access Scenario

    """

    name = "ps26_unauthorised_access"
    description = (
        "Unauthorised Access: Tests access control compliance. "
        "Evaluates whether models refuse unauthorized record access and credential misuse. "
    )
    tags = ["medical", "security", "compliance", "hipaa", "access_control"]

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
            self.data_path, msg=f"[PS26UnauthorisedAccessScenario] Required data file not found: '{self.data_path}'"
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
            display_name="PS26 Unauthorised Access",
            description=(
                "PacificMedSecurity v2.2 - Unauthorised Access: Evaluates model refusal of unauthorized access. "
                "Tests unauthorized record access, credential misuse, impersonation, and privilege escalation. "
                "Includes 739 instances across 4 harm levels."
            ),
            taxonomy=TaxonomyInfo(
                task="Compliance Evaluation",
                what="Determine if model helps with unauthorized medical record access",
                when="When evaluating HIPAA access control compliance",
                who="Healthcare staff, security officer, auditor",
                language="English",
            ),
            main_metric="ps26_unauthorised_access_accuracy",
            main_split="test",
        )
