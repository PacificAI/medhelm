"""PacificMedSecurity v2.2 - Consumer Health Data scenario.

Tests model compliance with consumer health privacy laws.
Evaluates whether models evade or comply with CCPA, CPRA, and state privacy requirements.
Regulatory basis: CCPA, CPRA, state privacy laws, FTC Health Breach Rule
"""

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.ps26_base_scenario import BasePS26Scenario
from helm.benchmark.scenarios.scenario import ScenarioMetadata


class PS26ConsumerHealthDataScenario(BasePS26Scenario):
    """
    PacificMedSecurity v2.2 - Consumer Health Data Scenario

    """

    name = "ps26_consumer_health_data"
    description = (
        "Consumer Health Data: Tests consumer privacy law compliance. "
        "Evaluates whether models refuse CCPA/CPRA/state law violations and data broker misuse. "
    )
    tags = ["medical", "privacy", "compliance", "ccpa", "consumer_health"]

    def get_metadata(self) -> ScenarioMetadata:
        """Return scenario metadata."""
        return ScenarioMetadata(
            name=self.name,
            display_name="PS26 Consumer Health Data",
            description=(
                "PacificMedSecurity v2.2 - Consumer Health Data: Evaluates model compliance with consumer "
                "health privacy laws (CCPA, CPRA, state laws). Tests privacy law violations, data broker "
                "misuse, and privacy policy circumvention. Includes 394 instances across 4 harm levels."
            ),
            taxonomy=TaxonomyInfo(
                task="Compliance Evaluation",
                what="Determine if model helps circumvent consumer health privacy laws",
                when="When evaluating non-HIPAA consumer health privacy compliance",
                who="Privacy officer, consumer app developer, data broker",
                language="English",
            ),
            main_metric="ps26_consumer_health_data_accuracy",
            main_split="test",
        )
