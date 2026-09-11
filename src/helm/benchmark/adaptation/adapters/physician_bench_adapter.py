"""Adapter that turns PhysicianBench instances into harness episode requests."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from helm.benchmark.adaptation.adapters.in_context_learning_adapter import InContextLearningAdapter
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.physician_bench_constants import (
    PB_HARNESS_DEPLOYMENT,
    PB_PROTOCOL,
)
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request


def parse_harness_knobs(instructions: str) -> Dict[str, Any]:
    if not instructions:
        return {}
    try:
        parsed = json.loads(instructions)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_pb_request(instance: Instance, adapter_spec) -> Request:
    """Encode the task envelope plus the evaluated MedHELM model onto a Request."""
    envelope = json.loads(instance.input.text)
    if envelope.get("pb_protocol") != PB_PROTOCOL:
        raise ValueError(
            f"PhysicianBench adapter expected protocol {PB_PROTOCOL}, "
            f"got {envelope.get('pb_protocol')!r}"
        )
    envelope["evaluated_model"] = adapter_spec.model
    envelope["evaluated_model_deployment"] = adapter_spec.model_deployment
    knobs = parse_harness_knobs(adapter_spec.instructions)
    for key in (
        "max_steps",
        "fhir_image",
        "port",
        "pb_root",
        "reasoning_effort",
        "temperature",
    ):
        value = knobs.get(key)
        if value not in (None, ""):
            envelope[key] = value
    return Request(
        model=adapter_spec.model,
        model_deployment=PB_HARNESS_DEPLOYMENT,
        prompt=json.dumps(envelope),
        num_completions=1,
        temperature=0.0,
        max_tokens=1,
        stop_sequences=[],
        random=adapter_spec.random,
    )


class PhysicianBenchAdapter(InContextLearningAdapter):
    """One Request per instance; the Client owns the FHIR + MiniAgent episode."""

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        del training_instances
        request = build_pb_request(eval_instance, self.adapter_spec)
        return [
            RequestState(
                instance=eval_instance,
                reference_index=None,
                request_mode=None,
                train_trial_index=train_trial_index,
                output_mapping=None,
                request=request,
                result=None,
                num_train_instances=0,
                prompt_truncated=False,
            )
        ]
