import re
from dataclasses import dataclass
from typing import Dict, List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric import MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG


_ANSWER_PATTERN = re.compile(r"(?<![A-Za-z])[A-G](?![A-Za-z])", re.IGNORECASE)


def parse_medrobustness_answer(response: str) -> str:
    """Return the single named answer letter, or X for an invalid response."""
    matches = _ANSWER_PATTERN.findall(response)
    return matches[0].upper() if len(matches) == 1 else "X"


@dataclass(frozen=True)
class _ArmResult:
    prediction: str
    correct_letter: str

    @property
    def correct(self) -> bool:
        return self.prediction == self.correct_letter

    @property
    def non_letter(self) -> bool:
        return self.prediction == "X"


class MedRobustnessMetric(EvaluateInstancesMetric):
    """Scores absolute accuracy and paired stability for PacificMedRobustness."""

    @staticmethod
    def _get_arm_result(request_state: RequestState) -> _ArmResult:
        correct_references = [
            reference for reference in request_state.instance.references if CORRECT_TAG in reference.tags
        ]
        if len(correct_references) != 1:
            raise ValueError(
                f"Expected exactly one correct reference for instance '{request_state.instance.id}', "
                f"found {len(correct_references)}"
            )

        completions = request_state.result.completions if request_state.result is not None else []
        if len(completions) > 1:
            raise ValueError(
                f"Expected at most one completion for instance '{request_state.instance.id}', found {len(completions)}"
            )
        prediction = parse_medrobustness_answer(completions[0].text) if completions else "X"
        return _ArmResult(
            prediction=prediction,
            correct_letter=correct_references[0].output.text.strip().upper(),
        )

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        del eval_cache_path
        pairs: Dict[str, Dict[str, _ArmResult]] = {}

        for request_state in request_states:
            extra_data = request_state.instance.extra_data
            if not extra_data:
                raise ValueError(f"Missing pairing metadata for instance '{request_state.instance.id}'")
            pair_id = str(extra_data.get("pair_id", "")).strip()
            arm = str(extra_data.get("arm", "")).strip().upper()
            if not pair_id or arm not in {"A", "B"}:
                raise ValueError(f"Invalid pairing metadata for instance '{request_state.instance.id}'")
            if arm in pairs.setdefault(pair_id, {}):
                raise ValueError(f"Duplicate arm {arm} for pair '{pair_id}'")
            pairs[pair_id][arm] = self._get_arm_result(request_state)

        stats = {
            name: Stat(MetricName(name))
            for name in [
                "medrobustness_accuracy",
                "medrobustness_arm_a_accuracy",
                "medrobustness_arm_b_accuracy",
                "medrobustness_non_letter_rate",
                "medrobustness_arm_a_non_letter_rate",
                "medrobustness_arm_b_non_letter_rate",
                "medrobustness_fully_correct_rate",
                "medrobustness_asymmetric_failure_rate",
                "medrobustness_failed_non_letter_rate",
                "medrobustness_recovery_rate",
                "medrobustness_letter_change_rate",
            ]
        }

        for pair_id, arms in pairs.items():
            if set(arms) != {"A", "B"}:
                raise ValueError(f"Pair '{pair_id}' must contain exactly arms A and B; found {sorted(arms)}")
            arm_a, arm_b = arms["A"], arms["B"]

            for arm, result in (("a", arm_a), ("b", arm_b)):
                stats["medrobustness_accuracy"].add(result.correct)
                stats[f"medrobustness_arm_{arm}_accuracy"].add(result.correct)
                stats["medrobustness_non_letter_rate"].add(result.non_letter)
                stats[f"medrobustness_arm_{arm}_non_letter_rate"].add(result.non_letter)

            stats["medrobustness_fully_correct_rate"].add(arm_a.correct and arm_b.correct)
            stats["medrobustness_letter_change_rate"].add(arm_a.prediction != arm_b.prediction)
            if arm_a.correct:
                stats["medrobustness_asymmetric_failure_rate"].add(not arm_b.correct and not arm_b.non_letter)
                stats["medrobustness_failed_non_letter_rate"].add(arm_b.non_letter)
            else:
                stats["medrobustness_recovery_rate"].add(arm_b.correct)

        return [stat for stat in stats.values() if stat.count > 0]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="medrobustness_accuracy",
                display_name="Overall accuracy",
                description="Accuracy across both arms.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_arm_a_accuracy",
                display_name="Arm A accuracy",
                description="Accuracy on the baseline arm.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_arm_b_accuracy",
                display_name="Arm B accuracy",
                description="Accuracy on the transformed arm.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_non_letter_rate",
                display_name="Non-letter rate",
                description="Rate of responses that do not identify exactly one answer from A to G.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_arm_a_non_letter_rate",
                display_name="Arm A non-letter rate",
                description="Non-letter response rate on the baseline arm.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_arm_b_non_letter_rate",
                display_name="Arm B non-letter rate",
                description="Non-letter response rate on the transformed arm.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_fully_correct_rate",
                display_name="Fully correct pair rate",
                description="Rate of pairs for which both arms are answered correctly.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_asymmetric_failure_rate",
                display_name="Asymmetric failure rate",
                description="Among pairs with a correct arm A, rate of valid but incorrect arm B answers.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_failed_non_letter_rate",
                display_name="Arm B non-letter failure rate",
                description="Among pairs with a correct arm A, rate of non-letter arm B responses.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_recovery_rate",
                display_name="Recovery rate",
                description="Among pairs with an incorrect arm A, rate for which arm B is correct.",
                lower_is_better=None,
            ),
            MetricMetadata(
                name="medrobustness_letter_change_rate",
                display_name="Letter change rate",
                description="Rate of pairs for which the predicted letter changes between arms.",
                lower_is_better=True,
            ),
        ]
