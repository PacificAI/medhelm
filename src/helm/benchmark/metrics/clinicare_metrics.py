"""CliniCARE metrics, read from the trial's reward.json carried in the CliniCAREClient completion."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# reward.json key -> MedHELM metric name. Keys are conditional: most appear only when measured.
OUTCOME_KEYS: Dict[str, str] = {
    "score": "clinicare_score",
    "outcome_label_parsed": "clinicare_verdict_parse_rate",
    "grounding_id_precision": "clinicare_grounding_precision",
    "policy_file_f1": "clinicare_policy_doc_f1",
    "evidence_calls": "clinicare_tool_calls",
    # Trust signals: recorded in stats.json, kept off the leaderboard by the schema.
    "evidence_calls_auth_error": "clinicare_evidence_calls_auth_error",
    "environment_degraded": "clinicare_environment_degraded",
    "report_present": "clinicare_report_present",
}

# LLM-judged, post-hoc (present only when the case was graded). Never blended into clinicare_score.
OFFLINE_KEYS: Dict[str, str] = {
    "process_pct": "clinicare_process_score",
    "policy_citation_support_rate": "clinicare_policy_support",
}

# reward.json stores process_pct as a percent (0-100) and every other key as a fraction (0-1).
# Scale it to a fraction so all CliniCARE columns share one scale.
SCALE: Dict[str, float] = {"process_pct": 0.01}


def _payload(request_state: RequestState) -> Optional[Dict[str, Any]]:
    result = request_state.result
    if result is None or not result.completions or not result.completions[0].text.strip():
        return None
    return json.loads(result.completions[0].text)


def _stats(reward: Optional[Dict[str, Any]], keys: Dict[str, str]) -> List[Stat]:
    if not reward:
        return []
    return [
        Stat(MetricName(name)).add(float(reward[key]) * SCALE.get(key, 1.0))
        for key, name in keys.items()
        if reward.get(key) is not None
    ]


class CliniCAREOutcomeMetric(Metric):
    """Deterministic scores from the in-container verifier.

    Never emits a Stat for an unmeasured key: a missing key is omitted, not counted as 0. A task
    with no scored trial (dead, timed out, never run) emits only ``clinicare_trial_failed`` = 1, so
    it shows up as a coverage gap rather than as a wrong answer.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        del adapter_spec, metric_service, eval_cache_path
        payload = _payload(request_state)
        failed = payload is None or payload.get("status") != "ok"
        return [Stat(MetricName("clinicare_trial_failed")).add(float(failed))] + _stats(
            (payload or {}).get("reward"), OUTCOME_KEYS
        )

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="clinicare_score",
                display_name="CliniCARE score",
                short_display_name="Score",
                description="Fraction of cases with the correct four-way verdict (in-container verifier).",
                lower_is_better=False,
                group="accuracy",
            ),
        ]


class CliniCAREOfflineMetric(Metric):
    """LLM-judged process and policy-citation scores. Emits nothing for a case that was not graded.

    Not an Annotator: MedHELM never runs these judges itself. They come from CliniCARE's own
    graders (run_case.py --grade, i.e. ``grade=true`` on the run entry).
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        del adapter_spec, metric_service, eval_cache_path
        payload = _payload(request_state) or {}
        stats = _stats(payload.get("reward"), OFFLINE_KEYS)
        if payload.get("grade_requested"):
            # Trust signal (off the leaderboard): grade=true was asked for but this case has no
            # judged scores, e.g. missing judge credentials. Only emitted when grading was requested.
            stats.append(Stat(MetricName("clinicare_grading_failed")).add(float(not payload.get("graded"))))
        return stats

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="clinicare_process_score",
                display_name="CliniCARE process score",
                short_display_name="Process",
                description="LLM-judged process rubric, 0-1 (did the investigation follow the scenario's steps).",
                lower_is_better=False,
                group="accuracy",
            ),
            MetricMetadata(
                name="clinicare_policy_support",
                display_name="CliniCARE policy citation support",
                short_display_name="Policy support",
                description="LLM-judged fraction of policy citations whose cited passage supports the claim.",
                lower_is_better=False,
                group="accuracy",
            ),
        ]
