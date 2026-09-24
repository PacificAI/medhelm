"""
Test cases for PP26 Proxy Access Scenario.

Tests whether LLMs can determine appropriate proxy access permissions to patient 
portal messages under HIPAA, state privacy laws, and clinical context.
"""

import pytest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd

from helm.benchmark.scenarios.pp26_proxy_access_scenario import PP26ProxyAccessScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


@pytest.mark.scenarios
class TestPP26ProxyAccessScenario:
    """Test cases for proxy access permissions to patient portal messages."""

    def test_proxy_access_scenario_initialization(self):
        """Test that the scenario initializes correctly."""
        scenario = PP26ProxyAccessScenario(data_path="/dummy/path.xlsx")
        assert scenario.name == "pp26_proxy_access"
        assert "proxy" in scenario.description.lower()

    def test_proxy_access_metadata(self):
        """Test that metadata is correctly configured."""
        scenario = PP26ProxyAccessScenario(data_path="/dummy/path.xlsx")
        metadata = scenario.get_metadata()
        assert metadata.name == "pp26_proxy_access"
        assert metadata.display_name == "ProxyAccess"
        assert metadata.main_metric == "exact_match"
        assert metadata.taxonomy.task == "Classification"

    def test_proxy_access_answer_choices(self):
        """Test that correct answer choices are defined."""
        assert PP26ProxyAccessScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B", "C", "D"]

    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.pd.read_excel")
    def test_proxy_access_send_to_proxy(self, mock_read_excel, mock_check_file):
        """Test critical case: authorized proxy should receive information."""
        data = {
            "question": [
                "Should the message be sent to the proxy account holder?",
            ],
            "message": [
                "Proxy: Parent of 14-year-old patient with documented authorization. Message: Appointment confirmation for routine physical exam.",
            ],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyAccessScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert len(instances[0].references) == 4
        assert instances[0].references[0].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.pd.read_excel")
    def test_proxy_access_deny_to_proxy(self, mock_read_excel, mock_check_file):
        """Test critical case: proxy should not receive confidential information."""
        data = {
            "question": [
                "Should the message be sent to the proxy account holder?",
            ],
            "message": [
                "Proxy: Parent of 17-year-old patient. Message: Results of STI test and prescriptions for contraception. Patient is emancipated.",
            ],
            "provided_label": ["B"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyAccessScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[1].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.pd.read_excel")
    def test_proxy_access_verify_identity(self, mock_read_excel, mock_check_file):
        """Test critical case: identity verification needed."""
        data = {
            "question": [
                "Should the message be sent to the proxy account holder?",
            ],
            "message": [
                "Proxy: Unverified identity claiming to be parent. Message: Appointment details and lab results.",
            ],
            "provided_label": ["C"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyAccessScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[2].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.pd.read_excel")
    def test_proxy_access_undecidable(self, mock_read_excel, mock_check_file):
        """Test critical case: decision undecidable from given facts."""
        data = {
            "question": [
                "Should the message be sent to the proxy account holder?",
            ],
            "message": [
                "Proxy: Account holder age unknown, authorization status unclear. Message: Clinical information of unspecified nature.",
            ],
            "provided_label": ["D"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyAccessScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[3].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_access_scenario.pd.read_excel")
    def test_proxy_access_minor_consent_consideration(self, mock_read_excel, mock_check_file):
        """Test edge case: state minor consent laws."""
        data = {
            "question": [
                "Should message be sent considering state minor consent law?",
            ],
            "message": [
                "Proxy: Parent of 16-year-old patient. State allows minors to consent to mental health treatment. Message: Mental health appointment reminder.",
            ],
            "provided_label": ["B"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyAccessScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
