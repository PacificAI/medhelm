import json

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.physician_bench_metrics import PhysicianBenchMetric
from helm.benchmark.scenarios.scenario import TEST_SPLIT, Input, Instance
from helm.common.request import GeneratedOutput, Request, RequestResult


def _request_state(payload: dict) -> RequestState:
    instance = Instance(input=Input(text=""), references=[], split=TEST_SPLIT, id="aortic_aneurysm_cad")
    return RequestState(
        instance=instance,
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping=None,
        request=Request(prompt="unused"),
        result=RequestResult(
            success=True,
            cached=False,
            completions=[GeneratedOutput(text=json.dumps(payload), logprob=0, tokens=[])],
            embedding=[],
        ),
        num_train_instances=0,
        prompt_truncated=False,
    )


def test_physician_bench_metric_parses_episode_json():
    payload = {
        "task_id": "aortic_aneurysm_cad",
        "passed": True,
        "score": 6.0,
        "max_points": 6.0,
        "percentage": 100.0,
        "steps": 44,
        "prompt_tokens": 12000,
        "completion_tokens": 3400,
        "tokens": 15400,
    }
    stats = {
        stat.name.name: stat.mean
        for stat in PhysicianBenchMetric().evaluate_generation(
            AdapterSpec(),
            _request_state(payload),
            None,  # type: ignore[arg-type]
            "",
        )
    }
    assert stats["physician_bench_score"] == 1.0
    assert stats["physician_bench_pass"] == 1.0
    assert stats["physician_bench_checkpoints_passed"] == 6.0
    assert stats["physician_bench_checkpoints_total"] == 6.0
    assert stats["physician_bench_steps"] == 44.0
    assert stats["physician_bench_prompt_tokens"] == 12000.0
    assert stats["physician_bench_completion_tokens"] == 3400.0
    assert stats["physician_bench_tokens"] == 15400.0


def test_physician_bench_metric_scores_checkpoint_fraction():
    """Native job 6/8 checkpoints -> physician_bench_score 0.75, pass 0."""
    payload = {
        "task_id": "postmenopausal_bleeding",
        "passed": False,
        "score": 6.0,
        "max_points": 8.0,
        "percentage": 75.0,
        "steps": 12,
    }
    stats = {
        stat.name.name: stat.mean
        for stat in PhysicianBenchMetric().evaluate_generation(
            AdapterSpec(),
            _request_state(payload),
            None,  # type: ignore[arg-type]
            "",
        )
    }
    assert stats["physician_bench_score"] == 0.75
    assert stats["physician_bench_pass"] == 0.0
    assert stats["physician_bench_checkpoints_passed"] == 6.0
    assert stats["physician_bench_checkpoints_total"] == 8.0


def test_physician_bench_metric_partial_checkpoints():
    payload = {
        "passed": False,
        "score": 3.0,
        "max_points": 6.0,
        "percentage": 50.0,
        "steps": 12,
    }
    stats = {
        stat.name.name: stat.mean
        for stat in PhysicianBenchMetric().evaluate_generation(
            AdapterSpec(),
            _request_state(payload),
            None,  # type: ignore[arg-type]
            "",
        )
    }
    assert stats["physician_bench_score"] == 0.5
    assert stats["physician_bench_pass"] == 0.0


def test_physician_bench_metric_sums_tokens_when_total_omitted():
    payload = {
        "passed": False,
        "score": 1.0,
        "max_points": 2.0,
        "percentage": 50.0,
        "steps": 2,
        "prompt_tokens": 100,
        "completion_tokens": 40,
    }
    stats = {
        stat.name.name: stat.mean
        for stat in PhysicianBenchMetric().evaluate_generation(
            AdapterSpec(),
            _request_state(payload),
            None,  # type: ignore[arg-type]
            "",
        )
    }
    assert stats["physician_bench_tokens"] == 140.0


def test_physician_bench_metric_zeros_on_empty_completion():
    instance = Instance(input=Input(text=""), references=[], split=TEST_SPLIT, id="x")
    request_state = RequestState(
        instance=instance,
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping=None,
        request=Request(prompt=""),
        result=RequestResult(success=False, cached=False, completions=[], embedding=[]),
        num_train_instances=0,
        prompt_truncated=False,
    )
    stats = {
        stat.name.name: stat.mean
        for stat in PhysicianBenchMetric().evaluate_generation(
            AdapterSpec(),
            request_state,
            None,  # type: ignore[arg-type]
            "",
        )
    }
    assert stats["physician_bench_score"] == 0.0
    assert stats["physician_bench_pass"] == 0.0
    assert stats["physician_bench_steps"] == 0.0
    assert stats["physician_bench_prompt_tokens"] == 0.0
    assert stats["physician_bench_completion_tokens"] == 0.0
    assert stats["physician_bench_tokens"] == 0.0
