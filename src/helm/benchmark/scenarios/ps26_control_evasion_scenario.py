"""PacificMedSecurity v2.2 - Control Evasion scenario.

Tests model compliance with HIPAA security rule safeguards.
Evaluates whether models evade or comply with access/audit/integrity/transmission controls.
Regulatory basis: HIPAA 45 CFR Part 164 Subpart C (Security Rule)
"""

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.ps26_base_scenario import BasePS26Scenario
from helm.benchmark.scenarios.scenario import ScenarioMetadata


class PS26ControlEvasionScenario(BasePS26Scenario):
    """
    PacificMedSecurity v2.2 - Control Evasion Scenario

    """

    name = "ps26_control_evasion"
    description = (
        "Control Evasion: Tests HIPAA security control compliance. "
        "Evaluates whether models provide guidance on proper controls or help evade them. "
    )
    tags = ["medical", "security", "compliance", "hipaa", "security_controls"]

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
