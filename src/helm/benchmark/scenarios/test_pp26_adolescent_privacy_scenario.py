"""
Test cases for PP26 Adolescent Privacy Scenario.

Tests whether LLMs can determine privacy rights and parental notification
requirements for adolescent patients under HIPAA and state-specific privacy laws.
"""

import pytest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd
from io import BytesIO

from helm.benchmark.scenarios.pp26_adolescent_privacy_scenario import PP26AdolescentPrivacyScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


@pytest.mark.scenarios
class TestPP26AdolescentPrivacyScenario:
    """Test cases for adolescent privacy rights and parental notification requirements."""

    def create_mock_adolescent_privacy_excel(self):
        """Create mock Excel data for adolescent privacy scenario."""
        data = {
            "question": [
                "Does the patient's progress note contain the patient's own information in a confidential category?",
                "Is mental health information from routine screening withholdable?",
                "Does the note contain information about family members' activities?",
            ],
            "note": [
                "Patient reports anxiety symptoms and has been prescribed sertraline. Discussed coping strategies.",
                "Routine mental health screening completed. No concerning findings noted. Patient reports good mood.",
                "Family history of diabetes. Mother has been managing her diabetes well.",
            ],
            "provided_label": ["A", "B", "A"],
        }
        df = pd.DataFrame(data)
        excel_file = BytesIO()
        df.to_excel(excel_file, sheet_name="Test cases", index=False)
        excel_file.seek(0)
        return excel_file

    def test_adolescent_privacy_scenario_initialization(self):
        """Test that the scenario initializes correctly."""
        scenario = PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx")
        assert scenario.name == "pp26_adolescent_privacy"
        assert "adolescent" in scenario.description.lower()

    def test_adolescent_privacy_metadata(self):
        """Test that metadata is correctly configured."""
        scenario = PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx")
        metadata = scenario.get_metadata()
        assert metadata.name == "pp26_adolescent_privacy"
        assert metadata.display_name == "AdolescentPrivacy"
        assert metadata.main_metric == "exact_match"
        assert metadata.main_split == "test"
        assert metadata.taxonomy.task == "Classification"

    def test_adolescent_privacy_answer_choices(self):
        """Test that correct answer choices are defined."""
        assert PP26AdolescentPrivacyScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]

    @patch("helm.benchmark.scenarios.pp26_adolescent_privacy_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_adolescent_privacy_scenario.pd.read_excel")
    def test_adolescent_privacy_instance_creation(self, mock_read_excel, mock_check_file):
        """Test that instances are created with correct structure."""
        data = {
            "question": [
                "Does the note contain information about the patient's sexual health that should be withheld?",
            ],
            "note": [
                "Patient counseled on contraceptive options including barrier methods and hormonal contraception.",
            ],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].split == TEST_SPLIT
        assert len(instances[0].references) == 2

        # Check that correct answer is tagged
        correct_refs = [ref for ref in instances[0].references if CORRECT_TAG in ref.tags]
        assert len(correct_refs) == 1
        assert correct_refs[0].output.text == "A"

    @patch("helm.benchmark.scenarios.pp26_adolescent_privacy_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_adolescent_privacy_scenario.pd.read_excel")
    def test_adolescent_privacy_withholding_logic(self, mock_read_excel, mock_check_file):
        """Test critical case: withholdable vs non-withholdable information."""
        data = {
            "question": [
                "Scenario 1: Confidential category",
                "Scenario 2: Non-confidential category",
            ],
            "note": [
                "Patient uses marijuana regularly. Started substance abuse counseling.",
                "Patient has been compliant with prescribed diabetes medication.",
            ],
            "provided_label": ["A", "B"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 2
        # First should be withholdable (A)
        assert instances[0].references[0].tags == [CORRECT_TAG]
        # Second should not be withholdable (B)
        assert instances[1].references[1].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_adolescent_privacy_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_adolescent_privacy_scenario.pd.read_excel")
    def test_adolescent_privacy_edge_case_family_safety(self, mock_read_excel, mock_check_file):
        """Test edge case: family safety information."""
        data = {
            "question": ["Does the note contain family safety information?"],
            "note": ["Discussed family conflict at home. No abuse reported."],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[0].tags == [CORRECT_TAG]
