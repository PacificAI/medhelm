import json
import threading
import time
from pathlib import Path

import pytest

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, ADAPT_PHYSICIAN_BENCH
from helm.benchmark.runner import execute_parallelism_for_run
from helm.benchmark.scenarios.physician_bench_constants import PB_HARNESS_DEPLOYMENT, PB_PROTOCOL
from helm.clients.physician_bench_client import PhysicianBenchClient, require_docker
from helm.common.request import Request


def test_execute_parallelism_for_pb_is_clamped_to_one():
    assert execute_parallelism_for_run(ADAPT_PHYSICIAN_BENCH, 4) == 1
    assert execute_parallelism_for_run(ADAPT_PHYSICIAN_BENCH, 1) == 1
    assert execute_parallelism_for_run(ADAPT_GENERATION, 4) == 4


def test_pb_client_serializes_concurrent_episodes(monkeypatch):
    client = PhysicianBenchClient()
    current = 0
    max_seen = 0
    lock = threading.Lock()
    start_together = threading.Barrier(2)

    def fake_run(self, envelope):
        nonlocal current, max_seen
        with lock:
            current += 1
            max_seen = max(max_seen, current)
        time.sleep(0.08)
        with lock:
            current -= 1
        return {
            "task_id": envelope.get("task_id"),
            "passed": False,
            "score": 0.0,
            "max_points": 0.0,
            "percentage": 0.0,
            "steps": 0,
        }

    monkeypatch.setattr(PhysicianBenchClient, "_run_episode", fake_run)
    prompt = json.dumps({"pb_protocol": PB_PROTOCOL, "task_id": "aortic_aneurysm_cad"})
    request = Request(prompt=prompt, model_deployment=PB_HARNESS_DEPLOYMENT)

    results: list = []

    def worker():
        start_together.wait(timeout=5)
        results.append(client.make_request(request))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert max_seen == 1
    assert len(results) == 2
    assert all(result.success for result in results)


def test_pb_client_stub_skips_run_task():
    client = PhysicianBenchClient()
    envelope = {
        "pb_protocol": PB_PROTOCOL,
        "task_id": "aortic_aneurysm_cad",
        "task_relpath": "tasks/v1/aortic_aneurysm_cad",
        "evaluated_model": "simple/model1",
        "evaluated_model_deployment": "simple/model1",
    }
    result = client.make_request(Request(prompt=json.dumps(envelope), model_deployment=PB_HARNESS_DEPLOYMENT))
    assert result.success
    payload = json.loads(result.completions[0].text)
    assert payload["passed"] is False
    assert payload["percentage"] == 0.0
    assert payload["error"] is None
    assert payload["pb_model_id"] == "stub"


def test_pb_client_unknown_model_fails_closed():
    client = PhysicianBenchClient()
    envelope = {
        "pb_protocol": PB_PROTOCOL,
        "task_id": "aortic_aneurysm_cad",
        "task_relpath": "tasks/v1/aortic_aneurysm_cad",
        "evaluated_model": "unknown/model",
        "evaluated_model_deployment": "unknown/model",
    }
    result = client.make_request(Request(prompt=json.dumps(envelope), model_deployment=PB_HARNESS_DEPLOYMENT))
    assert not result.success
    assert result.error_flags is not None
    assert not result.error_flags.is_fatal
    assert "no agent" in (result.error or "").lower() or "physician_bench_model_map" in (result.error or "")


def test_require_docker_fails_when_daemon_down(monkeypatch):
    class Result:
        returncode = 1
        stderr = "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?"
        stdout = ""

    monkeypatch.setattr(
        "helm.clients.physician_bench_client.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    with pytest.raises(RuntimeError, match="Docker daemon"):
        require_docker()


def test_require_docker_fails_when_cli_missing(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr("helm.clients.physician_bench_client.subprocess.run", boom)
    with pytest.raises(RuntimeError, match="docker CLI was not found"):
        require_docker()


def test_pb_client_stub_does_not_require_docker(monkeypatch):
    def boom():
        raise RuntimeError("docker should not be checked for stub")

    monkeypatch.setattr("helm.clients.physician_bench_client.require_docker", boom)
    client = PhysicianBenchClient()
    envelope = {
        "pb_protocol": PB_PROTOCOL,
        "task_id": "aortic_aneurysm_cad",
        "task_relpath": "tasks/v1/aortic_aneurysm_cad",
        "evaluated_model": "simple/model1",
        "evaluated_model_deployment": "simple/model1",
    }
    result = client.make_request(Request(prompt=json.dumps(envelope), model_deployment=PB_HARNESS_DEPLOYMENT))
    assert result.success


def test_pb_client_docker_down_is_fatal(monkeypatch):
    def boom():
        raise RuntimeError(
            "PhysicianBench requires a running Docker daemon to start the FHIR container. "
            "`docker info` failed: Cannot connect to the Docker daemon"
        )

    monkeypatch.setattr("helm.clients.physician_bench_client.require_docker", boom)
    monkeypatch.setattr(
        "helm.clients.physician_bench_client.resolve_pb_root",
        lambda *args, **kwargs: Path("/tmp/pb"),
    )
    client = PhysicianBenchClient()
    envelope = {
        "pb_protocol": PB_PROTOCOL,
        "task_id": "aortic_aneurysm_cad",
        "task_relpath": "tasks/v1/aortic_aneurysm_cad",
        "evaluated_model": "openai/gpt-4o-2024-05-13",
        "evaluated_model_deployment": "openai/gpt-4o-2024-05-13",
    }
    result = client.make_request(Request(prompt=json.dumps(envelope), model_deployment=PB_HARNESS_DEPLOYMENT))
    assert not result.success
    assert result.error_flags is not None
    assert result.error_flags.is_fatal
    assert "docker" in (result.error or "").lower()
