from typing import List, Optional

import pytest

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.medgrounding_metrics import MedGroundingMetric
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Input, Instance, Output, Reference
from helm.common.request import GeneratedOutput, Request, RequestResult


def _request_state(
    instance_id: str,
    key: str,
    prediction: str,
    extra_data: dict,
) -> RequestState:
    return RequestState(
        instance=Instance(
            id=instance_id,
            input=Input(text="prompt"),
            references=[Reference(output=Output(text=key), tags=[CORRECT_TAG])],
            split=TEST_SPLIT,
            extra_data=extra_data,
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


def _single(
    case_id: str,
    key: str,
    prediction: str,
    kept: Optional[str] = None,
    ascertain: str = "E",
    cannot_decide: str = "F",
) -> RequestState:
    extra_data = {
        "pair_id": case_id,
        "ascertain_letter": ascertain,
        "cannot_decide_letter": cannot_decide,
    }
    if kept is not None:
        extra_data["correct_letter_full_record"] = kept
    return _request_state(case_id, key, prediction, extra_data)


def _paired(case_id: str, arm: str, key: str, prediction: str) -> RequestState:
    return _request_state(
        f"{case_id}:{arm}",
        key,
        prediction,
        {"pair_id": case_id, "arm": arm, "ascertain_letter": "E", "cannot_decide_letter": "F"},
    )


def _means(stats: List[Stat]) -> dict[str, float]:
    return {stat.name.name: stat.mean for stat in stats if stat.mean is not None}


def test_single_record_stats() -> None:
    request_states = [
        _single("case-1", key="A", prediction="A", kept="B"),  # correct
        _single("case-2", key="A", prediction="B", kept="B"),  # kept
        _single("case-3", key="A", prediction="C", kept="B"),  # wrong move
        _single("case-4", key="A", prediction="E"),  # ascertain
        _single("case-5", key="A", prediction="no answer"),  # non-letter
        _single("case-6", key="C", prediction="F", kept="C"),  # cannot decide, kept not counted (kept == key)
    ]

    stats = _means(MedGroundingMetric().evaluate_instances(request_states, ""))

    assert stats["medgrounding_fully_correct_rate"] == pytest.approx(1 / 6)
    assert stats["medgrounding_non_letter_rate"] == pytest.approx(1 / 6)
    assert stats["medgrounding_kept_rate"] == pytest.approx(1 / 3)
    assert stats["medgrounding_wrong_move_rate"] == pytest.approx(1 / 6)
    assert stats["medgrounding_ascertain_rate"] == pytest.approx(1 / 6)
    assert stats["medgrounding_cannot_decide_rate"] == pytest.approx(1 / 6)


def test_single_record_without_full_record_key_skips_kept_rate() -> None:
    request_states = [
        _single("case-1", key="A", prediction="B"),
        _single("case-2", key="A", prediction="A"),
    ]

    stats = _means(MedGroundingMetric().evaluate_instances(request_states, ""))

    assert "medgrounding_kept_rate" not in stats
    assert stats["medgrounding_wrong_move_rate"] == pytest.approx(0.5)
    assert stats["medgrounding_fully_correct_rate"] == pytest.approx(0.5)


def test_paired_inversion_stats() -> None:
    request_states = [
        # Both arms correct: fully correct.
        _paired("pair-1", "A", key="A", prediction="A"),
        _paired("pair-1", "B", key="B", prediction="B"),
        # Same letter in both arms across a key change: kept, not fully correct.
        _paired("pair-2", "A", key="A", prediction="A"),
        _paired("pair-2", "B", key="B", prediction="A"),
        # Different letters in each arm: not kept.
        _paired("pair-3", "A", key="C", prediction="A"),
        _paired("pair-3", "B", key="C", prediction="C"),
    ]

    stats = _means(MedGroundingMetric().evaluate_instances(request_states, ""))

    assert stats["medgrounding_fully_correct_rate"] == pytest.approx(1 / 3)
    assert stats["medgrounding_arm_a_accuracy"] == pytest.approx(2 / 3)
    assert stats["medgrounding_arm_b_accuracy"] == pytest.approx(2 / 3)
    assert stats["medgrounding_kept_rate"] == pytest.approx(1 / 3)
    assert "medgrounding_wrong_move_rate" not in stats


def test_paired_ascertain_and_cannot_decide_rates() -> None:
    request_states = [
        _paired("pair-1", "A", key="A", prediction="E"),  # ascertain
        _paired("pair-1", "B", key="B", prediction="F"),  # cannot decide
        _paired("pair-2", "A", key="A", prediction="A"),
        _paired("pair-2", "B", key="B", prediction="B"),
    ]

    stats = _means(MedGroundingMetric().evaluate_instances(request_states, ""))

    assert stats["medgrounding_ascertain_rate"] == pytest.approx(1 / 4)
    assert stats["medgrounding_cannot_decide_rate"] == pytest.approx(1 / 4)


def test_paired_missing_arm_raises() -> None:
    request_states = [_paired("pair-1", "A", key="A", prediction="A")]

    with pytest.raises(ValueError, match="must contain exactly arms A and B"):
        MedGroundingMetric().evaluate_instances(request_states, "")


def test_duplicate_arm_raises() -> None:
    request_states = [
        _paired("pair-1", "A", key="A", prediction="A"),
        _paired("pair-1", "A", key="A", prediction="A"),
    ]

    with pytest.raises(ValueError, match="Duplicate arm"):
        MedGroundingMetric().evaluate_instances(request_states, "")


def test_missing_pair_id_raises() -> None:
    request_state = _request_state("case-1", key="A", prediction="A", extra_data={})

    with pytest.raises(ValueError, match="Missing pairing metadata"):
        MedGroundingMetric().evaluate_instances([request_state], "")
