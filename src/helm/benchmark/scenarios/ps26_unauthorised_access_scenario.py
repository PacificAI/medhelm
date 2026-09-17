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
    Output,
    Reference,
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
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "expected_behavior": row["expected_behavior"],
                        "harm_level": int(row["harm_level"]),
                        "scoring_rule": row["scoring_rule"],
                    }
                )
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
                references=[Reference(Output(text=row_data["expected_behavior"]), tags=[])],
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
