import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.metric import MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG


_ANSWER_PATTERN = re.compile(r"(?<![A-Za-z])[A-G](?![A-Za-z])", re.IGNORECASE)
# Explicit-marker patterns: "Answer: X", "answer is X", "the answer is X", "my answer: X", etc.
# Must not be followed by "or" or another letter to be considered explicit.
_EXPLICIT_MARKER_PATTERN = re.compile(
    r"(?:answer|choice|option)\s*(?:is|:)\s*([A-G])(?!\s*or|\s*[A-Za-z])", re.IGNORECASE
)
# Leading-answer patterns: "A.", "A)", "A:" at or near the start
_LEADING_ANSWER_PATTERN = re.compile(r"^\s*([A-G])[\.\):]", re.IGNORECASE)
SCORING_MODES = ("preserve", "invert", "control")


def parse_medrobustness_answer(response: str) -> str:
    """Return the single named answer letter, or X for an invalid response.

    Uses explicit-marker precedence:
    1. If response contains "answer is X" or similar explicit marker, use X.
    2. If response starts with "X." or "X)", use X.
    3. Otherwise, require exactly one standalone A-G letter in the entire response.
    4. If ambiguous or absent, return "X".
    """
    response = response.strip()

    # Check for explicit markers like "Answer: B" or "answer is C"
    marker_match = _EXPLICIT_MARKER_PATTERN.search(response)
    if marker_match:
        return marker_match.group(1).upper()

    # Check for leading answer like "A." or "B)"
    leading_match = _LEADING_ANSWER_PATTERN.match(response)
    if leading_match:
        return leading_match.group(1).upper()

    # Fall back to strict single-letter rule
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
    """Scores PacificMedRobustness pairs by their `scoring_mode`, following the pilot-3 scorer.

    - preserve: paired stability (both arms should give the same correct answer).
    - invert: per-arm accuracy and inversion rate (both arms correct).
    - control: noise floor, the rate at which a whitespace-only change flips a correct arm A answer.
    """

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

    @staticmethod
    def _get_pair_info(request_state: RequestState) -> Tuple[str, str, str]:
        extra_data = request_state.instance.extra_data
        if not extra_data:
            raise ValueError(f"Missing pairing metadata for instance '{request_state.instance.id}'")
        pair_id = str(extra_data.get("pair_id", "")).strip()
        arm = str(extra_data.get("arm", "")).strip().upper()
        scoring_mode = str(extra_data.get("scoring_mode", "preserve")).strip().lower()
        if not pair_id or arm not in {"A", "B"} or scoring_mode not in SCORING_MODES:
            raise ValueError(f"Invalid pairing metadata for instance '{request_state.instance.id}'")
        return pair_id, arm, scoring_mode

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        del eval_cache_path
        pairs: Dict[str, Dict[str, _ArmResult]] = {}
        pair_modes: Dict[str, str] = {}

        for request_state in request_states:
            pair_id, arm, scoring_mode = self._get_pair_info(request_state)
            if arm in pairs.setdefault(pair_id, {}):
                raise ValueError(f"Duplicate arm {arm} for pair '{pair_id}'")
            if pair_modes.setdefault(pair_id, scoring_mode) != scoring_mode:
                raise ValueError(f"Pair '{pair_id}' has conflicting scoring modes")
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
                "medrobustness_inverted_arm_a_accuracy",
                "medrobustness_inverted_arm_b_accuracy",
                "medrobustness_inversion_rate",
                "medrobustness_control_failure_rate",
            ]
        }

        for pair_id, arms in pairs.items():
            if set(arms) != {"A", "B"}:
                raise ValueError(f"Pair '{pair_id}' must contain exactly arms A and B; found {sorted(arms)}")
            arm_a, arm_b = arms["A"], arms["B"]
            mode = pair_modes[pair_id]

            if mode == "preserve":
                self._add_preserve_stats(stats, arm_a, arm_b)
            elif mode == "invert":
                stats["medrobustness_inverted_arm_a_accuracy"].add(arm_a.correct)
                stats["medrobustness_inverted_arm_b_accuracy"].add(arm_b.correct)
                stats["medrobustness_inversion_rate"].add(arm_a.correct and arm_b.correct)
            else:
                if arm_a.correct:
                    stats["medrobustness_control_failure_rate"].add(not arm_b.correct and not arm_b.non_letter)

        return [stat for stat in stats.values() if stat.count > 0]

    @staticmethod
    def _add_preserve_stats(stats: Dict[str, Stat], arm_a: _ArmResult, arm_b: _ArmResult) -> None:
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

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="medrobustness_accuracy",
                display_name="Overall accuracy",
                description="Accuracy across both arms of preserve pairs.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_arm_a_accuracy",
                display_name="Arm A accuracy",
                description="Accuracy on the baseline arm of preserve pairs.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_arm_b_accuracy",
                display_name="Arm B accuracy",
                description="Accuracy on the transformed arm of preserve pairs.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_non_letter_rate",
                display_name="Non-letter rate",
                description="Rate of preserve-pair responses that do not identify exactly one answer from A to G.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_arm_a_non_letter_rate",
                display_name="Arm A non-letter rate",
                description="Non-letter response rate on the baseline arm of preserve pairs.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_arm_b_non_letter_rate",
                display_name="Arm B non-letter rate",
                description="Non-letter response rate on the transformed arm of preserve pairs.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_fully_correct_rate",
                display_name="Fully correct pair rate",
                description="Rate of preserve pairs for which both arms are answered correctly.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_asymmetric_failure_rate",
                display_name="Asymmetric failure rate",
                description="Among preserve pairs with a correct arm A, rate of valid but incorrect arm B answers.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_failed_non_letter_rate",
                display_name="Arm B non-letter failure rate",
                description="Among preserve pairs with a correct arm A, rate of non-letter arm B responses.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_recovery_rate",
                display_name="Recovery rate",
                description="Among preserve pairs with an incorrect arm A, rate for which arm B is correct.",
                lower_is_better=None,
            ),
            MetricMetadata(
                name="medrobustness_letter_change_rate",
                display_name="Letter change rate",
                description="Rate of preserve pairs for which the predicted letter changes between arms.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medrobustness_inverted_arm_a_accuracy",
                display_name="Inverted pairs arm A accuracy",
                description="Accuracy on the baseline arm of invert pairs (opposite finding).",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_inverted_arm_b_accuracy",
                display_name="Inverted pairs arm B accuracy",
                description="Accuracy on the transformed arm of invert pairs (opposite finding).",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_inversion_rate",
                display_name="Inversion rate",
                description="Rate of invert pairs for which both arms are answered correctly.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medrobustness_control_failure_rate",
                display_name="Control failure rate",
                description=(
                    "Noise floor: among control pairs with a correct arm A, rate of valid but incorrect arm B "
                    "answers after whitespace-only changes."
                ),
                lower_is_better=True,
            ),
        ]
