"""
Integration tests for all PP26 healthcare privacy scenarios.

Tests cross-scenario validation and consistency across all PP26 privacy 
and compliance scenarios.
"""

import pytest

from helm.benchmark.scenarios.pp26_adolescent_privacy_scenario import PP26AdolescentPrivacyScenario
from helm.benchmark.scenarios.pp26_disclosure_basis_scenario import PP26DisclosureBasisScenario
from helm.benchmark.scenarios.pp26_proxy_access_scenario import PP26ProxyAccessScenario
from helm.benchmark.scenarios.pp26_proxy_leak_scenario import PP26ProxyLeakScenario


@pytest.mark.scenarios
class TestPP26ScenariosIntegration:
    """Integration tests across all PP26 scenarios."""

    def test_all_scenarios_have_correct_tags(self):
        """Verify all scenarios have proper taxonomy tags."""
        scenarios = [
            PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx"),
            PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyAccessScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyLeakScenario(data_path="/dummy/path.xlsx"),
        ]

        for scenario in scenarios:
            assert "biomedical" in scenario.tags
            assert "knowledge" in scenario.tags
            assert "reasoning" in scenario.tags

    def test_all_scenarios_have_correct_display_names(self):
        """Verify all scenarios have proper display names."""
        expected_names = {
            "pp26_adolescent_privacy": "AdolescentPrivacy",
            "pp26_disclosure_basis": "DisclosureBasis",
            "pp26_proxy_access": "ProxyAccess",
            "pp26_proxy_leak": "ProxyLeak",
        }

        scenarios = [
            PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx"),
            PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyAccessScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyLeakScenario(data_path="/dummy/path.xlsx"),
        ]

        for scenario in scenarios:
            metadata = scenario.get_metadata()
            assert metadata.display_name == expected_names[metadata.name]

    def test_all_scenarios_have_exact_match_metric(self):
        """Verify all scenarios use exact_match as main metric."""
        scenarios = [
            PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx"),
            PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyAccessScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyLeakScenario(data_path="/dummy/path.xlsx"),
        ]

        for scenario in scenarios:
            metadata = scenario.get_metadata()
            assert metadata.main_metric == "exact_match"

    def test_all_scenarios_use_classification_task(self):
        """Verify all scenarios are classification tasks."""
        scenarios = [
            PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx"),
            PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyAccessScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyLeakScenario(data_path="/dummy/path.xlsx"),
        ]

        for scenario in scenarios:
            metadata = scenario.get_metadata()
            assert metadata.taxonomy is not None
            assert metadata.taxonomy.task == "Classification"

    def test_all_scenarios_have_biomedical_context(self):
        """Verify all scenarios have biomedical content."""
        scenarios = [
            PP26AdolescentPrivacyScenario(data_path="/dummy/path.xlsx"),
            PP26DisclosureBasisScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyAccessScenario(data_path="/dummy/path.xlsx"),
            PP26ProxyLeakScenario(data_path="/dummy/path.xlsx"),
        ]

        for scenario in scenarios:
            has_biomedical_tag = "biomedical" in scenario.tags
            has_healthcare_keyword = any(
                keyword in scenario.description.lower()
                for keyword in ["health", "patient", "medical", "clinical", "hipaa", "privacy"]
            )
            assert has_biomedical_tag or has_healthcare_keyword
