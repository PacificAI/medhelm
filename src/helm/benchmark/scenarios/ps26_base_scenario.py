"""Base class for PacificMedSecurity v2.2 scenarios.

Provides common CSV loading and Instance creation logic.
Subclasses define scenario-specific metadata and tags.
"""

import sys
import csv
from typing import Any, Dict, List

from helm.benchmark.scenarios.scenario import (
    Input,
    Instance,
    Scenario,
    TEST_SPLIT,
)
from helm.common.general import check_file_exists

csv.field_size_limit(sys.maxsize)


class BasePS26Scenario(Scenario):
    """Base class for all PS26 scenarios.

    Handles:
    - CSV loading with schema validation
    - Instance creation from benchmark data
    - Common error handling and reporting

    Subclasses must define:
    - name: scenario identifier
    - description: short description
    - tags: classification tags
    """

    name: str  # Must be overridden by subclass
    description: str  # Must be overridden by subclass
    tags: List[str]  # Must be overridden by subclass

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def create_benchmark(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load benchmark data from CSV file with schema validation.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of dictionaries with validated benchmark data

        Raises:
            ValueError: If CSV is malformed or missing required columns
        """
        data: List[Dict[str, Any]] = []
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
        """Load instances from CSV file.

        Args:
            output_path: Output directory (required by parent class)

        Returns:
            List of Instance objects
        """
        check_file_exists(
            self.data_path, msg=f"[{self.__class__.__name__}] Required data file not found: '{self.data_path}'"
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
