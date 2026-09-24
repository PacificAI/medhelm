"""
Test cases for PP26 Proxy Leak Scenario.

Tests whether LLMs can identify privacy risks when proxy account holders access 
patient portal messages, including risks of unintended disclosure of sensitive 
health information.
"""

import pytest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd

from helm.benchmark.scenarios.pp26_proxy_leak_scenario import PP26ProxyLeakScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


@pytest.mark.scenarios
class TestPP26ProxyLeakScenario:
    """Test cases for privacy risks from proxy access to portal messages."""

    def test_proxy_leak_scenario_initialization(self):
        """Test that the scenario initializes correctly."""
        scenario = PP26ProxyLeakScenario(data_path="/dummy/path.xlsx")
        assert scenario.name == "pp26_proxy_leak"
        assert "leak" in scenario.description.lower() or "risk" in scenario.description.lower()

    def test_proxy_leak_metadata(self):
        """Test that metadata is correctly configured."""
        scenario = PP26ProxyLeakScenario(data_path="/dummy/path.xlsx")
        metadata = scenario.get_metadata()
        assert metadata.name == "pp26_proxy_leak"
        assert metadata.display_name == "ProxyLeak"
        assert metadata.main_metric == "exact_match"
        assert metadata.taxonomy.task == "Classification"

    def test_proxy_leak_answer_choices(self):
        """Test that correct answer choices are defined."""
        assert PP26ProxyLeakScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]

    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.pd.read_excel")
    def test_proxy_leak_confidential_disclosure(self, mock_read_excel, mock_check_file):
        """Test critical case: proxy access leaks confidential information."""
        data = {
            "question": [
                "Would a parent with proxy access learn confidential information?",
            ],
            "message": [
                "Message to adolescent: We received your STI test results and have prescribed treatment. Please schedule a follow-up appointment.",
            ],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyLeakScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert len(instances[0].references) == 2
        assert instances[0].references[0].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.pd.read_excel")
    def test_proxy_leak_no_confidential_disclosure(self, mock_read_excel, mock_check_file):
        """Test critical case: proxy access does not leak confidential information."""
        data = {
            "question": [
                "Would a parent with proxy access learn confidential information?",
            ],
            "message": [
                "Message to adolescent: Your appointment is confirmed for Tuesday at 2 PM. Please bring your insurance card and photo ID.",
            ],
            "provided_label": ["B"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyLeakScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[1].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.pd.read_excel")
    def test_proxy_leak_mental_health_disclosure(self, mock_read_excel, mock_check_file):
        """Test edge case: mental health information leakage."""
        data = {
            "question": [
                "Would mental health information be disclosed to proxy?",
            ],
            "message": [
                "Message to adolescent: Your antidepressant refill is ready for pickup. Please take as directed for your depression management.",
            ],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyLeakScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
        assert instances[0].references[0].tags == [CORRECT_TAG]

    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.pd.read_excel")
    def test_proxy_leak_substance_use_disclosure(self, mock_read_excel, mock_check_file):
        """Test edge case: substance use information leakage."""
        data = {
            "question": [
                "Would substance use information be disclosed to proxy?",
            ],
            "message": [
                "Message to adolescent: Your naltrexone prescription for alcohol use disorder has been renewed. Take one tablet daily.",
            ],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyLeakScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1

    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.pd.read_excel")
    def test_proxy_leak_reproductive_health_disclosure(self, mock_read_excel, mock_check_file):
        """Test edge case: reproductive health information leakage."""
        data = {
            "question": [
                "Would reproductive health information be disclosed?",
            ],
            "message": [
                "Message to adolescent: Your birth control prescription is ready for pickup at the pharmacy.",
            ],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyLeakScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1

    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.check_file_exists")
    @patch("helm.benchmark.scenarios.pp26_proxy_leak_scenario.pd.read_excel")
    def test_proxy_leak_family_safety_disclosure(self, mock_read_excel, mock_check_file):
        """Test edge case: family safety information leakage."""
        data = {
            "question": [
                "Would family safety information be disclosed?",
            ],
            "message": [
                "Message to adolescent: Following up on your recent report of domestic violence at home. We have resources available.",
            ],
            "provided_label": ["A"],
        }
        mock_read_excel.return_value = pd.DataFrame(data)

        scenario = PP26ProxyLeakScenario(data_path="/dummy/path.xlsx")
        with TemporaryDirectory() as tmpdir:
            instances = scenario.get_instances(tmpdir)

        assert len(instances) == 1
