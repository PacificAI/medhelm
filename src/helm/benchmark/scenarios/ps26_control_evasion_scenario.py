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
    Scenario,
    ScenarioMetadata,
    TEST_SPLIT,
)
from helm.common.general import check_file_exists

csv.field_size_limit(sys.maxsize)


class PS26ControlEvasionScenario(Scenario):
    """
    PacificMedSecurity v2.2 - Control Evasion Scenario

    """

    name = "ps26_control_evasion"
    description = (
        "Control Evasion: Tests HIPAA security control compliance. "
        "Evaluates whether models provide guidance on proper controls or help evade them. "
    )
    tags = ["medical", "security", "compliance", "hipaa", "security_controls"]

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
            self.data_path, msg=f"[PS26ControlEvasionScenario] Required data file not found: '{self.data_path}'"
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
