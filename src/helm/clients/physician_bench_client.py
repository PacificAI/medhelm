"""PhysicianBench Client: one HELM Request is one FHIR + MiniAgent + pytest episode."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

# The FHIR host port is machine-global. Episode cwd, env, and imports run in a child process.
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


def _run_task_in_subprocess(pb_root: Path, task_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Run PhysicianBench ``run_task`` in a child interpreter.

    The child loads the checkout ``.env``, ``sys.path`` entry, and ``scripts.run_task``
    module. Those changes die with the child, so a later checkout cannot reuse them.
    """
    fd, name = tempfile.mkstemp(prefix="pb-episode-", suffix=".json")
    os.close(fd)
    result_path = Path(name)
    payload = {
        "pb_root": str(pb_root),
        "result_path": str(result_path),
        "task_kwargs": task_kwargs,
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "helm.clients.physician_bench_worker"],
            input=json.dumps(payload),
            text=True,
            cwd=str(pb_root),
        )
        raw = result_path.read_text(encoding="utf-8") if result_path.is_file() else ""
        if not raw.strip():
            raise RuntimeError(f"PhysicianBench episode worker exited {completed.returncode} without a result")
        try:
            report = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"PhysicianBench episode worker exited {completed.returncode} with an unreadable result"
            ) from exc
        if not isinstance(report, dict) or not report.get("ok") or completed.returncode != 0:
            error = str(report.get("error") or "") if isinstance(report, dict) else ""
            raise RuntimeError(error or f"PhysicianBench episode worker exited {completed.returncode}")
        result = report.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("PhysicianBench episode worker returned a non-object result")
        return result
    finally:
        result_path.unlink(missing_ok=True)


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
    """Runs PhysicianBench `run_task` in a child process. Does not cache FHIR episodes."""

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
        except Exception as exc:  # noqa: BLE001 - most episode failures must not abort the suite
            hwarn(f"PhysicianBench episode failed: {exc}")
            return self._failed_result(
                str(exc),
                envelope,
                started,
                fatal=is_environment_error(str(exc)),
            )

        error = payload.get("error")
        if not error:
            return self._completed_result(payload, started)

        # run_task can return scores, usage, and job_dir together with an error.
        # Keep that payload. A non-fatal error stays success=True so the executor
        # does not replace the completion with an empty string.
        message = str(error)
        hwarn(f"PhysicianBench episode failed: {message}")
        if is_environment_error(message):
            return self._failed_result(message, envelope, started, fatal=True, payload=payload)
        return self._completed_result(payload, started, error=message)

    def _completed_result(
        self,
        payload: Dict[str, Any],
        started: float,
        error: Optional[str] = None,
    ) -> RequestResult:
        text = json.dumps(payload)
        return RequestResult(
            success=True,
            cached=False,
            error=error,
            request_time=time.time() - started,
            request_datetime=int(time.time()),
            completions=[GeneratedOutput(text=text, logprob=0, tokens=[Token(text=text, logprob=0)])],
            embedding=[],
        )

    def _failed_result(
        self,
        error: str,
        envelope: Dict[str, Any],
        started: float,
        fatal: bool = False,
        payload: Optional[Dict[str, Any]] = None,
    ) -> RequestResult:
        if payload is None:
            body: Dict[str, Any] = _empty_completion_payload(error, envelope)
        else:
            body = dict(payload)
            body["error"] = error
        text = json.dumps(body)
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
        if max_steps is None or max_steps == "":
            max_steps = DEFAULT_MAX_STEPS
        else:
            max_steps = int(str(max_steps))

        port = envelope.get("port")
        if port is None or port == "":
            port = DEFAULT_PORT
        else:
            port = int(str(port))

        fhir_image = envelope.get("fhir_image") or DEFAULT_FHIR_IMAGE

        reasoning_effort = envelope.get("reasoning_effort")
        if reasoning_effort == "":
            reasoning_effort = None

        temperature = envelope.get("temperature")
        if temperature is None or temperature == "":
            temperature = None
        else:
            temperature = float(str(temperature))

        require_docker()

        hlog(
            f"PhysicianBench episode task={envelope.get('task_id')} "
            f"evaluated_model={evaluated_model} backend={backend} "
            f"pb_model={pb_model_id} max_steps={max_steps} port={port}"
        )

        result = _run_task_in_subprocess(
            pb_root,
            {
                "task_folder": str(task_path),
                "model": str(pb_model_id),
                "max_steps": int(max_steps),
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "fhir_image": str(fhir_image),
                "port": int(port),
            },
        )

        payload = dict(result)
        payload.update(extra)
        payload.setdefault("error", None)
        return payload
