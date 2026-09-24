"""
Test cases for PP26 Disclosure Basis Scenario.

Tests whether LLMs can determine the legal basis for disclosure of patient 
health information under HIPAA, 42 CFR Part 2, and state privacy laws.
"""

import pytest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd

from helm.benchmark.scenarios.pp26_disclosure_basis_scenario import PP26DisclosureBasisScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


@pytest.mark.scenarios
class TestPP26DisclosureBasisScenario:
    """Test cases for legal basis of patient information disclosure."""

    def test_disclosure_basis_scenario_initialization(self):
        """Test that the scenario initializes correctly."""
        scenario = PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx")
        assert scenario.name == "pp26_disclosure_basis"
        assert "disclosure" in scenario.description.lower()

    def test_disclosure_basis_metadata(self):
        """Test that metadata is correctly configured."""
        scenario = PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx")
        metadata = scenario.get_metadata()
        assert metadata.name == "pp26_disclosure_basis"
        assert metadata.display_name == "DisclosureBasis"
        assert metadata.main_metric == "exact_match"
        assert metadata.main_split == "test"
        assert metadata.taxonomy.task == "Classification"

    def test_disclosure_basis_answer_choices(self):
        """Test that correct answer choices are defined."""
        assert PP26DisclosureBasisScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B", "C"]

    @patch("helm.benchmark.scenarios.pp26_disclosure_basis_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_disclosure_basis_scenario.pd.read_excel")
    def test_disclosure_basis_permitted_disclosure(self, mock_read_excel, mock_check_file):
        """Test critical case: permitted disclosure under HIPAA."""
        data = {
            "question": [
                "Can health information be disclosed to a court with a valid subpoena?",
            ],
            "scenario": [
                "A healthcare provider received a subpoena duces tecum from a court requiring disclosure of patient medical records.",
            ],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert len(instances[0].references) == 3
        assert instances[0].references[0].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_disclosure_basis_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_disclosure_basis_scenario.pd.read_excel")
    def test_disclosure_basis_not_permitted(self, mock_read_excel, mock_check_file):
        """Test critical case: disclosure not permitted."""
        data = {
            "question": [
                "Can a covered entity disclose patient information to a third party without authorization?",
            ],
            "scenario": [
                "A marketing company requests patient medical records to perform epidemiological research. No patient authorization provided.",
            ],
            "provided_label": ["B"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[1].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_disclosure_basis_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_disclosure_basis_scenario.pd.read_excel")
    def test_disclosure_basis_undecidable(self, mock_read_excel, mock_check_file):
        """Test critical case: disclosure legality is undecidable from given facts."""
        data = {
            "question": [
                "Can information be disclosed based on applicable law?",
            ],
            "scenario": [
                "A healthcare provider receives a request to disclose patient information. The authorization status and applicable law are unclear.",
            ],
            "provided_label": ["C"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[2].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_disclosure_basis_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_disclosure_basis_scenario.pd.read_excel")
    def test_disclosure_basis_42_cfr_part_2(self, mock_read_excel, mock_check_file):
        """Test edge case: 42 CFR Part 2 substance use treatment records."""
        data = {
            "question": [
                "Can substance use treatment records be disclosed to a third party?",
            ],
            "scenario": [
                "Patient is enrolled in a substance abuse treatment program covered by 42 CFR Part 2. A request for disclosure is received without proper authorization.",
            ],
            "provided_label": ["B"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[1].tags == [CORRECT_TAG]
