"""PacificMedSecurity v2.2 - Breach Response scenario.

Tests model compliance with HIPAA breach notification requirements.
Evaluates whether models evade or comply with breach notification procedures.
Regulatory basis: HIPAA 45 CFR 164.402-414 (Notification)
"""

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.ps26_base_scenario import BasePS26Scenario
from helm.benchmark.scenarios.scenario import ScenarioMetadata


class PS26BreachResponseScenario(BasePS26Scenario):
    """
    PacificMedSecurity v2.2 - Breach Response Scenario

    """

    name = "ps26_breach_response"
    description = (
        "Breach Response: Tests HIPAA breach notification compliance. "
        "Evaluates whether models provide guidance on legitimate breach procedures (accept) or "
        "help evade notification requirements (refuse)."
    )
    tags = ["medical", "security", "compliance", "hipaa", "breach_notification"]

    def get_metadata(self) -> ScenarioMetadata:
        """Return scenario metadata."""
        return ScenarioMetadata(
            name=self.name,
            display_name="PS26 Breach Response",
            description=(
                "PacificMedSecurity v2.2 - Breach Response: Evaluates model compliance with "
                "HIPAA breach notification requirements (45 CFR 164.402-414). Tests 4 key areas: "
                "breach thresholds, notification procedures, notice content, and documentation retention. "
                "Includes 396 instances across 4 harm levels (0-3)."
            ),
            taxonomy=TaxonomyInfo(
                task="Compliance Evaluation",
                what="Determine if model evades or complies with HIPAA breach notification procedures",
                when="When evaluating healthcare security compliance",
                who="Healthcare administrator, security officer, regulator",
                language="English",
            ),
            main_metric="ps26_breach_response_accuracy",
            main_split="test",
        )
