import json
from typing import Any, Dict, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.clinicare_metrics import CliniCAREOfflineMetric, CliniCAREOutcomeMetric
from helm.benchmark.scenarios.scenario import TEST_SPLIT, Input, Instance
from helm.common.request import GeneratedOutput, Request, RequestResult


def _state(payload: Optional[Dict[str, Any]]) -> RequestState:
    completions = [GeneratedOutput(text=json.dumps(payload), logprob=0.0, tokens=[])] if payload else []
    return RequestState(
        instance=Instance(input=Input(text=""), references=[], split=TEST_SPLIT, id="advanced-imaging-000001"),
        reference_index=None,
        request_mode=None,
        train_trial_index=0,
        output_mapping=None,
        request=Request(model="openai/gpt-x", prompt="{}"),
        result=RequestResult(success=payload is not None, embedding=[], completions=completions, cached=False),
        num_train_instances=0,
        prompt_truncated=False,
    )


def _means(metric, state: RequestState) -> Dict[str, float]:
    stats = metric.evaluate_generation(AdapterSpec(), state, None, "")
    return {stat.name.name: stat.mean for stat in stats}


def test_missing_key_emits_no_stat_rather_than_zero():
    reward = {"score": 1.0, "outcome_label_parsed": 1, "evidence_calls": 5}  # no grounding_id_precision
    means = _means(CliniCAREOutcomeMetric(), _state({"status": "ok", "reward": reward}))
    assert means["clinicare_score"] == 1.0
    assert means["clinicare_tool_calls"] == 5.0
    assert means["clinicare_trial_failed"] == 0.0
    assert "clinicare_grounding_precision" not in means


def test_null_value_emits_no_stat():
    means = _means(CliniCAREOutcomeMetric(), _state({"status": "ok", "reward": {"score": 0.0, "policy_file_f1": None}}))
    assert means["clinicare_score"] == 0.0
    assert "clinicare_policy_doc_f1" not in means


def test_dead_trial_counts_as_failed_not_as_a_wrong_answer():
    means = _means(CliniCAREOutcomeMetric(), _state(None))
    assert means == {"clinicare_trial_failed": 1.0}


def test_agent_error_is_scored_but_flagged():
    reward = {"score": 0.0, "report_present": 0}
    means = _means(CliniCAREOutcomeMetric(), _state({"status": "agent_error", "reward": reward}))
    assert means["clinicare_score"] == 0.0
    assert means["clinicare_trial_failed"] == 1.0


def test_offline_metric_silent_without_grading():
    assert _means(CliniCAREOfflineMetric(), _state({"status": "ok", "reward": {"score": 1.0}})) == {}


def test_offline_metric_reads_graded_keys_on_one_scale():
    # process_pct is a percent in reward.json; the metric reports it as a fraction like the rest.
    reward = {"score": 1.0, "process_pct": 75.0, "policy_citation_support_rate": 0.5}
    means = _means(CliniCAREOfflineMetric(), _state({"status": "ok", "reward": reward}))
    assert means == {"clinicare_process_score": 0.75, "clinicare_policy_support": 0.5}


def test_grading_failed_only_reported_when_grading_was_requested():
    metric = CliniCAREOfflineMetric()
    reward = {"score": 1.0}
    assert "clinicare_grading_failed" not in _means(metric, _state({"status": "ok", "reward": reward}))
    ungraded = _state({"status": "ok", "reward": reward, "grade_requested": True, "graded": False})
    assert _means(metric, ungraded) == {"clinicare_grading_failed": 1.0}
    graded_reward = {"score": 1.0, "process_pct": 75.0, "policy_citation_support_rate": 0.5}
    graded = _state({"status": "ok", "reward": graded_reward, "grade_requested": True, "graded": True})
    assert _means(metric, graded) == {
        "clinicare_process_score": 0.75,
        "clinicare_policy_support": 0.5,
        "clinicare_grading_failed": 0.0,
    }
