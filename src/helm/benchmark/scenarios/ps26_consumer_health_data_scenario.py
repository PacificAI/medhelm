"""PacificMedSecurity v2.2 - Consumer Health Data scenario.

Tests model compliance with consumer health privacy laws.
Evaluates whether models evade or comply with CCPA, CPRA, and state privacy requirements.
Regulatory basis: CCPA, CPRA, state privacy laws, FTC Health Breach Rule
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


class PS26ConsumerHealthDataScenario(Scenario):
    """
    PacificMedSecurity v2.2 - Consumer Health Data Scenario

    """

    name = "ps26_consumer_health_data"
    description = (
        "Consumer Health Data: Tests consumer privacy law compliance. "
        "Evaluates whether models refuse CCPA/CPRA/state law violations and data broker misuse. "
    )
    tags = ["medical", "privacy", "compliance", "ccpa", "consumer_health"]

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
            self.data_path, msg=f"[PS26ConsumerHealthDataScenario] Required data file not found: '{self.data_path}'"
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
            display_name="PS26 Consumer Health Data",
            description=(
                "PacificMedSecurity v2.2 - Consumer Health Data: Evaluates model compliance with consumer "
                "health privacy laws (CCPA, CPRA, state laws). Tests privacy law violations, data broker "
                "misuse, and privacy policy circumvention. Includes 394 instances across 4 harm levels."
            ),
            taxonomy=TaxonomyInfo(
                task="Compliance Evaluation",
                what="Determine if model helps circumvent consumer health privacy laws",
                when="When evaluating non-HIPAA consumer health privacy compliance",
                who="Privacy officer, consumer app developer, data broker",
                language="English",
            ),
            main_metric="ps26_consumer_health_data_accuracy",
            main_split="test",
        )
