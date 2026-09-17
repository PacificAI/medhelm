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
    Output,
    Reference,
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
            self.data_path, msg=f"[PS26DataTheftScenario] Required data file not found: '{self.data_path}'"
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
