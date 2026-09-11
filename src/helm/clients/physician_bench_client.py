"""PhysicianBench Client: one HELM Request is one FHIR + MiniAgent + pytest episode."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

from helm.benchmark.scenarios.physician_bench_constants import (
    DEFAULT_FHIR_IMAGE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PORT,
    PB_HARNESS_DEPLOYMENT,
    PB_PROTOCOL,
    PB_ROOT_ENV,
)
from helm.benchmark.scenarios.physician_bench_scenario import resolve_pb_root
from helm.clients.client import Client
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.request import ErrorFlags, GeneratedOutput, Request, RequestResult, Token

# Docker host port, FHIR_BASE_URL, and os.chdir are process-global.
_PB_EPISODE_LOCK = threading.Lock()
_DOCKER_ERROR_MARKERS = (
    "docker",
    "fhir container failed",
    "fhir server",
    "cannot connect to the docker daemon",
)


def require_docker() -> None:
    """Fail closed if the Docker daemon is not running. FHIR lives in a container."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "PhysicianBench requires Docker to start the FHIR server, but the docker CLI was not found."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "PhysicianBench Docker preflight timed out (`docker info`). Is the Docker daemon running?"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or f"exit {result.returncode}"
        raise RuntimeError(
            "PhysicianBench requires a running Docker daemon to start the FHIR container. "
            f"`docker info` failed: {detail}"
        )


def is_environment_error(error: str) -> bool:
    """True for Docker / FHIR-server failures that should abort the suite."""
    text = error.lower()
    return any(marker in text for marker in _DOCKER_ERROR_MARKERS)


def load_model_map() -> Dict[str, Any]:
    map_path = Path(__file__).resolve().parents[1] / "benchmark" / "static" / "physician_bench_model_map.yaml"
    with map_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def lookup_model_mapping(evaluated_model: str, evaluated_deployment: str = "") -> Optional[Dict[str, Any]]:
    """Return a map row, or None if the model is unknown (fail closed)."""
    data = load_model_map()
    entries = data.get("models") or []
    exact = None
    prefix_match = None
    prefix_len = -1
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if evaluated_deployment and entry.get("model_deployment") == evaluated_deployment:
            return entry
        if entry.get("model") == evaluated_model:
            exact = entry
            continue
        prefix = entry.get("model_prefix")
        if prefix and evaluated_model.startswith(prefix) and len(prefix) > prefix_len:
            prefix_match = entry
            prefix_len = len(prefix)
    if exact is not None:
        return exact
    if prefix_match is not None:
        return prefix_match
    return None


@contextmanager
def _pb_runtime(pb_root: Path) -> Iterator[None]:
    prev_cwd = Path.cwd()
    env_file = pb_root / ".env"
    if str(pb_root) not in sys.path:
        sys.path.insert(0, str(pb_root))
    os.chdir(pb_root)
    try:
        from dotenv import load_dotenv

        if env_file.is_file():
            load_dotenv(env_file, override=False)
        yield
    finally:
        os.chdir(prev_cwd)


def _empty_completion_payload(error: str, envelope: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": envelope.get("task_id"),
        "passed": False,
        "score": 0.0,
        "max_points": 0.0,
        "percentage": 0.0,
        "steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tokens": 0,
        "job_dir": envelope.get("job_dir") or "",
        "evaluated_model": envelope.get("evaluated_model"),
        "evaluated_model_deployment": envelope.get("evaluated_model_deployment"),
        "pb_model_id": None,
        "error": error,
    }


class PhysicianBenchClient(Client):
    """Runs PhysicianBench `run_task` in-process. Does not cache FHIR episodes."""

    def __init__(
        self,
        cache_config: Optional[CacheConfig] = None,
        model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
    ):
        del cache_config, tokenizer_name
        self.model_name = model_name or PB_HARNESS_DEPLOYMENT

    def make_request(self, request: Request) -> RequestResult:
        started = time.time()
        try:
            envelope = json.loads(request.prompt)
        except json.JSONDecodeError as exc:
            return self._failed_result(str(exc), {}, started)

        if envelope.get("pb_protocol") != PB_PROTOCOL:
            return self._failed_result(
                f"Expected pb_protocol={PB_PROTOCOL}",
                envelope,
                started,
            )

        try:
            with _PB_EPISODE_LOCK:
                payload = self._run_episode(envelope)
            error = payload.get("error")
            if error:
                raise RuntimeError(str(error))
            text = json.dumps(payload)
            return RequestResult(
                success=True,
                cached=False,
                request_time=time.time() - started,
                request_datetime=int(time.time()),
                completions=[GeneratedOutput(text=text, logprob=0, tokens=[Token(text=text, logprob=0)])],
                embedding=[],
            )
        except Exception as exc:  # noqa: BLE001 - most episode failures must not abort the suite
            hwarn(f"PhysicianBench episode failed: {exc}")
            return self._failed_result(
                str(exc),
                envelope,
                started,
                fatal=is_environment_error(str(exc)),
            )

    def _failed_result(
        self,
        error: str,
        envelope: Dict[str, Any],
        started: float,
        fatal: bool = False,
    ) -> RequestResult:
        text = json.dumps(_empty_completion_payload(error, envelope))
        return RequestResult(
            success=False,
            cached=False,
            error=error,
            error_flags=ErrorFlags(is_retriable=False, is_fatal=fatal),
            request_time=time.time() - started,
            request_datetime=int(time.time()),
            completions=[GeneratedOutput(text=text, logprob=0, tokens=[Token(text=text, logprob=0)])],
            embedding=[],
        )

    def _resolve_backend(self, evaluated_model: str, evaluated_deployment: str) -> tuple[str, Dict[str, Any]]:
        if evaluated_deployment == PB_HARNESS_DEPLOYMENT:
            raise ValueError("evaluated_model_deployment must not be pb/harness")
        mapping = lookup_model_mapping(evaluated_model, evaluated_deployment)
        if mapping is None:
            raise ValueError(
                f"PhysicianBench has no agent for model {evaluated_model!r} "
                f"(deployment {evaluated_deployment!r}); add a map entry in "
                "physician_bench_model_map.yaml."
            )
        backend = mapping.get("backend") or ""
        if backend not in ("stub", "physician_bench"):
            raise ValueError(f"Unsupported PhysicianBench backend: {backend!r}")
        return backend, mapping

    def _run_episode(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        evaluated_model = str(envelope.get("evaluated_model") or "")
        evaluated_deployment = str(envelope.get("evaluated_model_deployment") or "")
        backend, mapping = self._resolve_backend(evaluated_model, evaluated_deployment)
        pb_model_id = mapping.get("pb_model_id") or evaluated_model

        extra = {
            "evaluated_model": evaluated_model,
            "evaluated_model_deployment": evaluated_deployment,
            "pb_model_id": pb_model_id,
        }

        if backend == "stub":
            hlog(
                f"PhysicianBench stub episode (no Docker, no MiniAgent) "
                f"task={envelope.get('task_id')} evaluated_model={evaluated_model}"
            )
            payload = _empty_completion_payload("", envelope)
            payload["error"] = None
            payload.update(extra)
            return payload

        pb_root = resolve_pb_root(str(envelope.get("pb_root") or os.environ.get(PB_ROOT_ENV) or ""))

        task_relpath = envelope.get("task_relpath")
        if not task_relpath:
            raise ValueError("PhysicianBench envelope missing task_relpath")
        task_path = pb_root / task_relpath

        max_steps = envelope.get("max_steps")
        if max_steps not in (None, ""):
            max_steps = int(max_steps)
        else:
            max_steps = DEFAULT_MAX_STEPS
        port = envelope.get("port")
        port = int(port) if port not in (None, "") else DEFAULT_PORT
        fhir_image = envelope.get("fhir_image") or DEFAULT_FHIR_IMAGE
        reasoning_effort = envelope.get("reasoning_effort") or None
        if reasoning_effort == "":
            reasoning_effort = None
        temperature = envelope.get("temperature")
        if temperature in (None, ""):
            temperature = None
        else:
            temperature = float(temperature)

        require_docker()

        hlog(
            f"PhysicianBench episode task={envelope.get('task_id')} "
            f"evaluated_model={evaluated_model} backend={backend} "
            f"pb_model={pb_model_id} max_steps={max_steps} port={port}"
        )

        with _pb_runtime(pb_root):
            from scripts.run_task import run_task

            result = run_task(
                task_folder=str(task_path),
                model=str(pb_model_id),
                max_steps=int(max_steps),
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                fhir_image=str(fhir_image),
                port=int(port),
            )

        payload = dict(result)
        payload.update(extra)
        payload.setdefault("error", None)
        return payload
