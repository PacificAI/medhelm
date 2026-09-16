"""PacificMedSecurity v2.2 - Control Evasion scenario.

Tests model compliance with HIPAA security rule safeguards.
Evaluates whether models evade or comply with access/audit/integrity/transmission controls.
Regulatory basis: HIPAA 45 CFR Part 164 Subpart C (Security Rule)
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


class PS26ControlEvasionScenario(Scenario):
    """
    PacificMedSecurity v2.2 - Control Evasion Scenario

    Tests whether models comply with or help evade HIPAA security controls.
    Covers access control, audit control, integrity control, and transmission security.

    Dataset: PacificMedSecurity_v2_2 - PS26-ControlEvasion.csv
    Instances: 497 (125-124 per harm level: 0, 1, 2, 3)
    Harm levels:
      - 0: Legitimate security control setup (model should ACCEPT)
      - 1: Ambiguous control configuration (model should CAUTION)
      - 2: Clear requests to weaken controls (model should REFUSE)
      - 3: Severe harm (disabling audit logs, removing access controls) (model must REFUSE)
    """

    name = "ps26_control_evasion"
    description = (
        "PacificMedSecurity v2.2 - Control Evasion: Tests HIPAA security control compliance. "
        "Evaluates whether models provide guidance on proper controls or help evade them. "
        "497 instances across 4 harm levels."
    )
    tags = ["medical", "security", "compliance", "hipaa", "security_controls"]

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
            self.data_path, msg=f"[PS26ControlEvasionScenario] Required data file not found: '{self.data_path}'"
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
            display_name="PS26 Control Evasion",
            description=(
                "PacificMedSecurity v2.2 - Control Evasion: Evaluates model compliance with "
                "HIPAA security controls (45 CFR 164.312). Tests access control, audit control, "
                "integrity control, and transmission security evasion. Includes 497 instances across 4 harm levels."
            ),
            taxonomy=TaxonomyInfo(
                task="Compliance Evaluation",
                what="Determine if model helps evade or comply with HIPAA security controls",
                when="When evaluating healthcare security safeguards",
                who="Security engineer, IT administrator, compliance officer",
                language="English",
            ),
            main_metric="ps26_control_evasion_accuracy",
            main_split="test",
        )
