from dataclasses import dataclass
from typing import Dict, List, Optional

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_instances_metric import EvaluateInstancesMetric
from helm.benchmark.metrics.medrobustness_metrics import parse_medrobustness_answer
from helm.benchmark.metrics.metric import MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG


@dataclass(frozen=True)
class _Answer:
    prediction: str
    key: str
    ascertain_letter: str
    cannot_decide_letter: str
    kept_letter: Optional[str]

    @property
    def correct(self) -> bool:
        return self.prediction == self.key

    @property
    def non_letter(self) -> bool:
        return self.prediction == "X"


class MedGroundingMetric(EvaluateInstancesMetric):
    """Scores PacificMedGrounding benchmarks.

    - Single-record benchmarks: each case is one instance. Fully correct means the key was chosen.
      Kept means the model keeps the answer the full record supports when that differs from the key.
      Wrong move means a valid letter that is neither the key, the kept answer, nor ascertain/cannot-decide.
    - Paired benchmarks (Inversion): both arms must be correct. Kept means arm B keeps the arm A key
      when the two keys differ.
    """

    @staticmethod
    def _get_answer(request_state: RequestState) -> _Answer:
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

        extra_data = request_state.instance.extra_data or {}
        kept_letter = extra_data.get("correct_letter_full_record")
        return _Answer(
            prediction=prediction,
            key=correct_references[0].output.text.strip().upper(),
            ascertain_letter=str(extra_data.get("ascertain_letter", "")).strip().upper(),
            cannot_decide_letter=str(extra_data.get("cannot_decide_letter", "")).strip().upper(),
            kept_letter=str(kept_letter).strip().upper() if kept_letter else None,
        )

    @staticmethod
    def _get_pair_info(request_state: RequestState) -> tuple:
        extra_data = request_state.instance.extra_data
        if not extra_data or not str(extra_data.get("pair_id", "")).strip():
            raise ValueError(f"Missing pairing metadata for instance '{request_state.instance.id}'")
        pair_id = str(extra_data["pair_id"]).strip()
        arm = str(extra_data.get("arm", "")).strip().upper()
        return pair_id, arm

    def evaluate_instances(self, request_states: List[RequestState], eval_cache_path: str) -> List[Stat]:
        del eval_cache_path
        pairs: Dict[str, Dict[str, _Answer]] = {}
        for request_state in request_states:
            pair_id, arm = self._get_pair_info(request_state)
            arms = pairs.setdefault(pair_id, {})
            if arm in arms:
                raise ValueError(f"Duplicate arm '{arm}' for pair '{pair_id}'")
            arms[arm] = self._get_answer(request_state)

        stats = {
            name: Stat(MetricName(name))
            for name in [
                "medgrounding_fully_correct_rate",
                "medgrounding_arm_a_accuracy",
                "medgrounding_arm_b_accuracy",
                "medgrounding_non_letter_rate",
                "medgrounding_kept_rate",
                "medgrounding_wrong_move_rate",
                "medgrounding_ascertain_rate",
                "medgrounding_cannot_decide_rate",
            ]
        }

        for pair_id, arms in pairs.items():
            if set(arms) == {""}:
                self._add_single_stats(stats, arms[""])
            elif set(arms) == {"A", "B"}:
                self._add_paired_stats(stats, arms["A"], arms["B"])
            else:
                raise ValueError(f"Pair '{pair_id}' must contain exactly arms A and B; found {sorted(arms)}")

        return [stat for stat in stats.values() if stat.count > 0]

    @staticmethod
    def _add_single_stats(stats: Dict[str, Stat], answer: _Answer) -> None:
        stats["medgrounding_fully_correct_rate"].add(answer.correct)
        stats["medgrounding_non_letter_rate"].add(answer.non_letter)
        stats["medgrounding_ascertain_rate"].add(answer.prediction == answer.ascertain_letter)
        stats["medgrounding_cannot_decide_rate"].add(answer.prediction == answer.cannot_decide_letter)

        if answer.kept_letter is not None and answer.kept_letter != answer.key:
            stats["medgrounding_kept_rate"].add(answer.prediction == answer.kept_letter)

        is_wrong_move = (
            not answer.correct
            and not answer.non_letter
            and answer.prediction not in (answer.ascertain_letter, answer.cannot_decide_letter, answer.kept_letter)
        )
        stats["medgrounding_wrong_move_rate"].add(is_wrong_move)

    @staticmethod
    def _add_paired_stats(stats: Dict[str, Stat], arm_a: _Answer, arm_b: _Answer) -> None:
        stats["medgrounding_fully_correct_rate"].add(arm_a.correct and arm_b.correct)
        stats["medgrounding_arm_a_accuracy"].add(arm_a.correct)
        stats["medgrounding_arm_b_accuracy"].add(arm_b.correct)
        for arm in (arm_a, arm_b):
            stats["medgrounding_non_letter_rate"].add(arm.non_letter)
            stats["medgrounding_ascertain_rate"].add(arm.prediction == arm.ascertain_letter)
            stats["medgrounding_cannot_decide_rate"].add(arm.prediction == arm.cannot_decide_letter)

        # Kept: arm B keeps arm A's key when the keys differ (arm B should have answered differently)
        if arm_a.key != arm_b.key:
            stats["medgrounding_kept_rate"].add(arm_b.prediction == arm_a.key)

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="medgrounding_fully_correct_rate",
                display_name="Fully correct rate",
                description=(
                    "Rate of cases answered with the key. For paired benchmarks, both arms must be answered correctly."
                ),
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medgrounding_arm_a_accuracy",
                display_name="Arm A accuracy",
                description="Accuracy on the baseline arm of paired benchmarks.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medgrounding_arm_b_accuracy",
                display_name="Arm B accuracy",
                description="Accuracy on the transformed arm of paired benchmarks.",
                lower_is_better=False,
            ),
            MetricMetadata(
                name="medgrounding_non_letter_rate",
                display_name="Non-letter rate",
                description="Rate of responses that do not identify exactly one answer from A to G.",
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medgrounding_kept_rate",
                display_name="Kept answer rate",
                description=(
                    "Among cases whose full-record answer differs from the key, rate at which the model keeps the "
                    "full-record answer after the necessary fact is removed or inverted."
                ),
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medgrounding_wrong_move_rate",
                display_name="Wrong move rate",
                description=(
                    "Rate of valid answers that are neither the key, the kept answer, ascertain, nor cannot-decide."
                ),
                lower_is_better=True,
            ),
            MetricMetadata(
                name="medgrounding_ascertain_rate",
                display_name="Ascertain rate",
                description="Rate of answers that choose the ascertain option.",
                lower_is_better=None,
            ),
            MetricMetadata(
                name="medgrounding_cannot_decide_rate",
                display_name="Cannot decide rate",
                description="Rate of answers that choose the cannot-decide option.",
                lower_is_better=None,
            ),
        ]
