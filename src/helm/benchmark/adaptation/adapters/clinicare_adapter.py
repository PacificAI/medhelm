"""Adapter that turns CliniCARE instances into requests for the CliniCARE client."""

from __future__ import annotations

import json
from typing import List

from helm.benchmark.adaptation.adapters.in_context_learning_adapter import InContextLearningAdapter
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.clinicare_constants import CLINICARE_PROTOCOL, CLINICARE_HARNESS_DEPLOYMENT
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request


def build_clinicare_request(instance: Instance, adapter_spec) -> Request:
    """Encode the task id, the evaluated model and the run knobs onto a Request.

    The task id is the lookup key (never the prompt text). ``model`` stays the evaluated model so the
    leaderboard row is shared with every other MedHELM scenario; only the deployment is rerouted.
    """
    knobs = json.loads(adapter_spec.instructions) if adapter_spec.instructions else {}
    envelope = {
        "clinicare_protocol": CLINICARE_PROTOCOL,
        "task_id": (instance.extra_data or {})["task_id"],
        "evaluated_model": adapter_spec.model,
        "evaluated_model_deployment": adapter_spec.model_deployment,
        **knobs,
    }
    return Request(
        model=adapter_spec.model,
        model_deployment=CLINICARE_HARNESS_DEPLOYMENT,
        prompt=json.dumps(envelope),
        num_completions=1,
        temperature=0.0,
        max_tokens=1,
        stop_sequences=[],
        random=adapter_spec.random,
    )


class CliniCAREAdapter(InContextLearningAdapter):
    """One Request per instance; the client owns the agent episode."""

    def generate_requests(
        self, eval_instance: Instance, train_trial_index: int, training_instances: List[Instance]
    ) -> List[RequestState]:
        del training_instances
        return [
            RequestState(
                instance=eval_instance,
                reference_index=None,
                request_mode=None,
                train_trial_index=train_trial_index,
                output_mapping=None,
                request=build_clinicare_request(eval_instance, self.adapter_spec),
                result=None,
                num_train_instances=0,
                prompt_truncated=False,
            )
        ]
