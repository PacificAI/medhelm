from typing import List

import pytest

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.medrobustness_metrics import MedRobustnessMetric, parse_medrobustness_answer
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Input, Instance, Output, Reference
from helm.common.request import GeneratedOutput, Request, RequestResult


def _request_state(pair_id: str, arm: str, correct: str, prediction: str) -> RequestState:
    return RequestState(
        instance=Instance(
            id=f"{pair_id}:{arm}",
            input=Input(text="prompt"),
            references=[Reference(output=Output(text=correct), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
            extra_data={"pair_id": pair_id, "arm": arm},
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
