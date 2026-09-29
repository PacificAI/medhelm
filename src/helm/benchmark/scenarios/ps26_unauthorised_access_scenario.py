"""PacificMedSecurity v2.2 - Unauthorised Access scenario.

Tests model compliance with access control requirements.
Evaluates whether models refuse unauthorized record access and credential misuse.
Regulatory basis: HIPAA 45 CFR 164.308-312 (Access controls)
"""

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.ps26_base_scenario import BasePS26Scenario
from helm.benchmark.scenarios.scenario import ScenarioMetadata


class PS26UnauthorisedAccessScenario(BasePS26Scenario):
    """
    PacificMedSecurity v2.2 - Unauthorised Access Scenario

    """

    name = "ps26_unauthorised_access"
    description = (
        "Unauthorised Access: Tests access control compliance. "
        "Evaluates whether models refuse unauthorized record access and credential misuse. "
    )
    tags = ["medical", "security", "compliance", "hipaa", "access_control"]

    def get_metadata(self) -> ScenarioMetadata:
        """Return scenario metadata."""
        return ScenarioMetadata(
            name=self.name,
            display_name="PS26 Unauthorised Access",
            description=(
                "PacificMedSecurity v2.2 - Unauthorised Access: Evaluates model refusal of unauthorized access. "
                "Tests unauthorized record access, credential misuse, impersonation, and privilege escalation. "
                "Includes 739 instances across 4 harm levels."
            ),
            taxonomy=TaxonomyInfo(
                task="Compliance Evaluation",
                what="Determine if model helps with unauthorized medical record access",
                when="When evaluating HIPAA access control compliance",
                who="Healthcare staff, security officer, auditor",
                language="English",
            ),
            main_metric="ps26_unauthorised_access_accuracy",
            main_split="test",
        )
