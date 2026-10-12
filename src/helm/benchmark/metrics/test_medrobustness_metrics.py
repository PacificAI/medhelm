from typing import List

import pytest

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.medrobustness_metrics import MedRobustnessMetric, parse_medrobustness_answer
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Input, Instance, Output, Reference
from helm.common.request import GeneratedOutput, Request, RequestResult


def _request_state(
    pair_id: str, arm: str, correct: str, prediction: str, scoring_mode: str = "preserve"
) -> RequestState:
    return RequestState(
        instance=Instance(
            id=f"{pair_id}:{arm}",
            input=Input(text="prompt"),
            references=[Reference(output=Output(text=correct), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
            extra_data={"pair_id": pair_id, "arm": arm, "scoring_mode": scoring_mode},
        ),
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping=None,
        request=Request(prompt="prompt"),
        result=RequestResult(
            success=True,
            cached=False,
            completions=[GeneratedOutput(text=prediction, logprob=0.0, tokens=[])],
            embedding=[],
        ),
        num_train_instances=0,
        prompt_truncated=False,
    )


def _means(stats: List[Stat]) -> dict[str, float]:
    return {stat.name.name: stat.mean for stat in stats if stat.mean is not None}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A", "A"),
        ("Answer: c.", "C"),
        ("A or B", "X"),
        ("no answer", "X"),
        ("A A", "X"),
    ],
)
def test_parse_medrobustness_answer(response: str, expected: str) -> None:
    assert parse_medrobustness_answer(response) == expected


def test_evaluate_instances_computes_paired_metrics() -> None:
    request_states = [
        _request_state("pair-1", "A", "A", "A"),
        _request_state("pair-1", "B", "A", "A"),
        _request_state("pair-2", "A", "B", "B"),
        _request_state("pair-2", "B", "B", "C"),
        _request_state("pair-3", "A", "C", "A"),
        _request_state("pair-3", "B", "C", "C"),
        _request_state("pair-4", "A", "D", "D"),
        _request_state("pair-4", "B", "D", "no answer"),
    ]

    stats = _means(MedRobustnessMetric().evaluate_instances(request_states, ""))

    assert stats["medrobustness_accuracy"] == pytest.approx(0.625)
    assert stats["medrobustness_arm_a_accuracy"] == pytest.approx(0.75)
    assert stats["medrobustness_arm_b_accuracy"] == pytest.approx(0.5)
    assert stats["medrobustness_fully_correct_rate"] == pytest.approx(0.25)
    assert stats["medrobustness_asymmetric_failure_rate"] == pytest.approx(1 / 3)
    assert stats["medrobustness_failed_non_letter_rate"] == pytest.approx(1 / 3)
    assert stats["medrobustness_recovery_rate"] == pytest.approx(1.0)
    assert stats["medrobustness_letter_change_rate"] == pytest.approx(0.75)
    assert stats["medrobustness_non_letter_rate"] == pytest.approx(0.125)


def test_evaluate_instances_rejects_incomplete_pair() -> None:
    with pytest.raises(ValueError, match="must contain exactly arms A and B"):
        MedRobustnessMetric().evaluate_instances([_request_state("pair-1", "A", "A", "A")], "")


def test_invert_pairs_are_scored_per_arm_and_kept_out_of_stability() -> None:
    request_states = [
        _request_state("inv-1", "A", "A", "A", "invert"),
        _request_state("inv-1", "B", "A", "A", "invert"),
        _request_state("inv-2", "A", "B", "B", "invert"),
        _request_state("inv-2", "B", "B", "C", "invert"),
        _request_state("pair-1", "A", "A", "A"),
        _request_state("pair-1", "B", "A", "B"),
    ]

    stats = _means(MedRobustnessMetric().evaluate_instances(request_states, ""))

    assert stats["medrobustness_inverted_arm_a_accuracy"] == pytest.approx(1.0)
    assert stats["medrobustness_inverted_arm_b_accuracy"] == pytest.approx(0.5)
    assert stats["medrobustness_inversion_rate"] == pytest.approx(0.5)
    assert stats["medrobustness_fully_correct_rate"] == pytest.approx(0.0)
    assert stats["medrobustness_asymmetric_failure_rate"] == pytest.approx(1.0)
    assert stats["medrobustness_accuracy"] == pytest.approx(0.5)


def test_control_pairs_give_noise_floor_only() -> None:
    request_states = [
        _request_state("ctl-1", "A", "A", "A", "control"),
        _request_state("ctl-1", "B", "A", "B", "control"),
        _request_state("ctl-2", "A", "B", "B", "control"),
        _request_state("ctl-2", "B", "B", "B", "control"),
        _request_state("ctl-3", "A", "C", "A", "control"),
        _request_state("ctl-3", "B", "C", "no answer", "control"),
    ]

    stats = _means(MedRobustnessMetric().evaluate_instances(request_states, ""))

    assert stats["medrobustness_control_failure_rate"] == pytest.approx(0.5)
    assert set(stats) == {"medrobustness_control_failure_rate"}


def test_evaluate_instances_rejects_conflicting_modes_within_pair() -> None:
    request_states = [
        _request_state("pair-1", "A", "A", "A", "preserve"),
        _request_state("pair-1", "B", "A", "A", "invert"),
    ]
    with pytest.raises(ValueError, match="conflicting scoring modes"):
        MedRobustnessMetric().evaluate_instances(request_states, "")


def test_evaluate_instances_rejects_unknown_scoring_mode() -> None:
    with pytest.raises(ValueError, match="Invalid pairing metadata"):
        MedRobustnessMetric().evaluate_instances(
            [_request_state("pair-1", "A", "A", "A", "abstain"), _request_state("pair-1", "B", "A", "A", "abstain")],
            "",
        )
