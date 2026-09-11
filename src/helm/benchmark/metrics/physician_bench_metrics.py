"""Metrics for PhysicianBench episode scores."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.common.hierarchical_logger import hlog


def parse_pb_completion(request_state: RequestState) -> Optional[Dict[str, Any]]:
    if request_state.result is None or not request_state.result.completions:
        return None
    text = request_state.result.completions[0].text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        hlog("Warning: PhysicianBench completion was not valid JSON")
        return None
    return payload if isinstance(payload, dict) else None


class PhysicianBenchMetric(Metric):
    """Unpack PhysicianBench score JSON from the episode completion."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        del adapter_spec, metric_service, eval_cache_path
        payload = parse_pb_completion(request_state)
        if not payload:
            hlog("Warning: No PhysicianBench completion payload; recording zeros")
            payload = {}

        percentage = float(payload.get("percentage") or 0.0)
        score = float(payload.get("score") or 0.0)
        max_points = float(payload.get("max_points") or 0.0)
        passed = 1.0 if payload.get("passed") else 0.0
        steps = float(payload.get("steps") or 0.0)
        prompt_tokens = float(payload.get("prompt_tokens") or 0.0)
        completion_tokens = float(payload.get("completion_tokens") or 0.0)
        tokens = float(payload.get("tokens") or (prompt_tokens + completion_tokens))
        main = percentage / 100.0 if percentage else (score / max_points if max_points > 0 else 0.0)

        return [
            Stat(MetricName("physician_bench_score")).add(main),
            Stat(MetricName("physician_bench_pass")).add(passed),
            Stat(MetricName("physician_bench_checkpoints_passed")).add(score),
            Stat(MetricName("physician_bench_checkpoints_total")).add(max_points),
            Stat(MetricName("physician_bench_steps")).add(steps),
            Stat(MetricName("physician_bench_prompt_tokens")).add(prompt_tokens),
            Stat(MetricName("physician_bench_completion_tokens")).add(completion_tokens),
            Stat(MetricName("physician_bench_tokens")).add(tokens),
        ]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="physician_bench_score",
                display_name="PhysicianBench Score",
                short_display_name="PB Score",
                description="Fraction of PhysicianBench pytest checkpoints passed, normalized to 0–1.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="physician_bench_pass",
                display_name="PhysicianBench Pass Rate",
                short_display_name="PB Pass",
                description="1 if every pytest checkpoint on the task passed, else 0.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="physician_bench_checkpoints_passed",
                display_name="PhysicianBench Checkpoints Passed",
                short_display_name="PB Passed",
                description="Number of pytest checkpoints passed on the task.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="physician_bench_checkpoints_total",
                display_name="PhysicianBench Checkpoints Total",
                short_display_name="PB Total",
                description="Number of pytest checkpoints on the task.",
                lower_is_better=False,
                group=None,
            ),
            MetricMetadata(
                name="physician_bench_steps",
                display_name="PhysicianBench Steps",
                short_display_name="PB Steps",
                description="Number of MiniAgent LLM steps (trajectory llm_response events).",
                lower_is_better=True,
                group=None,
            ),
            MetricMetadata(
                name="physician_bench_prompt_tokens",
                display_name="PhysicianBench Prompt Tokens",
                short_display_name="PB Prompt Tok",
                description="Sum of MiniAgent prompt tokens across all llm_response steps on the task.",
                lower_is_better=True,
                group=None,
            ),
            MetricMetadata(
                name="physician_bench_completion_tokens",
                display_name="PhysicianBench Completion Tokens",
                short_display_name="PB Completion Tok",
                description="Sum of MiniAgent completion tokens across all llm_response steps on the task.",
                lower_is_better=True,
                group=None,
            ),
            MetricMetadata(
                name="physician_bench_tokens",
                display_name="PhysicianBench Tokens",
                short_display_name="PB Tokens",
                description="Total MiniAgent tokens on the task (prompt + completion, summed across steps).",
                lower_is_better=True,
                group=None,
            ),
        ]
