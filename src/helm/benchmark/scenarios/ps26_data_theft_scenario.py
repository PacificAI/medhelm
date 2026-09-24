"""PacificMedSecurity v2.2 - Data Theft scenario.

Tests model resistance to data theft, ransomware, and trafficking requests.
Evaluates whether models refuse requests for exfiltration, ransomware, and concealment.
Regulatory basis: HIPAA, criminal law, breach prevention
"""

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.ps26_base_scenario import BasePS26Scenario
from helm.benchmark.scenarios.scenario import ScenarioMetadata


class PS26DataTheftScenario(BasePS26Scenario):
    """
    PacificMedSecurity v2.2 - Data Theft Scenario

    """

    name = "ps26_data_theft"
    description = (
        "Data Theft: Tests refusal of data theft and ransomware requests. "
        "Evaluates whether models refuse exfiltration, ransomware, trafficking, and concealment assistance. "
    )
    tags = ["medical", "security", "compliance", "data_theft", "ransomware"]

    def get_metadata(self) -> ScenarioMetadata:
        """Return scenario metadata."""
        return ScenarioMetadata(
            name=self.name,
            display_name="PS26 Data Theft",
            description=(
                "PacificMedSecurity v2.2 - Data Theft: Evaluates model refusal of data theft requests. "
                "Tests exfiltration techniques, ransomware attacks, health information trafficking, "
                "and attack concealment. Includes 544 instances across 4 harm levels with focus on attacker/insider perspectives."
            ),
            taxonomy=TaxonomyInfo(
                task="Compliance Evaluation",
                what="Determine if model assists with data theft, ransomware, or trafficking",
                when="When evaluating model security against criminal uses",
                who="Attacker, insider, security researcher",
                language="English",
            ),
            main_metric="ps26_data_theft_accuracy",
            main_split="test",
        )
