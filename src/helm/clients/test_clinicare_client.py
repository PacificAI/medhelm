import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from helm.benchmark.scenarios.clinicare_constants import CLINICARE_PROTOCOL, CLINICARE_RUN_CASE
from helm.clients import clinicare_client
from helm.clients.clinicare_client import CliniCAREClient, job_name, read_trial, task_checksum
from helm.common.request import Request
from helm.proxy.retry import NonRetriableException

# Placeholder ids only: real CliniCARE task ids identify patients.
TASK = "advanced-imaging-000001"
OTHER = "sepsis-bundle-000002"
MODEL = "openai/gpt-x"
ROW = {"model": MODEL, "system": "codex-gptx", "harbor_model": MODEL, "effort": None}
# dirhash(<a task dir from _tasks>, "sha256"), computed with the dirhash package harbor uses.
TASK_CHECKSUM = "8045fec8ddafb8abb1b6d037ccc8807eeee5102fbc6ba39323b032249967c20c"


def _trial(
    job: Path,
    task_id: str,
    *,
    model: str = MODEL,
    verified: bool = True,
    exc: Optional[str] = None,
    reward: Optional[Dict[str, Any]] = None,
    checksum: Optional[str] = TASK_CHECKSUM,
) -> Path:
    trial = job / f"{task_id[:32]}__abcdefg"
    (trial / "verifier").mkdir(parents=True)
    result = {
        "task_id": {"path": f"/x/benchmark/tasks/{task_id}"},
        "task_checksum": checksum,
        "config": {"agent": {"name": "pkg:Agent", "model_name": model}},
        "agent_info": {"version": "1.0"},
        "verifier_result": {"rewards": {"reward": 1.0}} if verified else None,
        "exception_info": {"exception_type": exc, "exception_message": "boom"} if exc else None,
    }
    (trial / "result.json").write_text(json.dumps(result))
    if verified:
        (trial / "verifier" / "reward.json").write_text(json.dumps(reward or {"score": 1.0, "evidence_calls": 3}))
        (trial / "verifier" / "report.md").write_text("placeholder report")
    return trial


def _tasks(root: Path, *task_ids: str) -> Path:
    tasks = root / "benchmark" / "tasks"
    for task_id in task_ids:
        (tasks / task_id / "environment").mkdir(parents=True)
        (tasks / task_id / "empty").mkdir()  # dirhash skips empty dirs
        (tasks / task_id / "instruction.md").write_text("placeholder task")
        (tasks / task_id / "environment" / "Dockerfile").write_text("FROM scratch\n")
    return tasks


def _request(**knobs: Any) -> Request:
    envelope = {"clinicare_protocol": CLINICARE_PROTOCOL, "task_id": TASK, "evaluated_model": MODEL, **knobs}
    return Request(model=MODEL, model_deployment="clinicare/harness", prompt=json.dumps(envelope))


@pytest.fixture(autouse=True)
def _fresh_client_state(monkeypatch):
    # CliniCAREClient keeps run-wide state on the class (shared across the instances AutoClient may
    # build); give every test a clean run.
    monkeypatch.setattr(CliniCAREClient, "_replay_index", {})
    monkeypatch.setattr(CliniCAREClient, "_docker_checked", False)
    monkeypatch.setattr(CliniCAREClient, "_warned", set())


@pytest.fixture(autouse=True)
def _one_row_map(monkeypatch):
    monkeypatch.setattr(clinicare_client, "load_model_map", lambda: {"models": [dict(ROW)]})


@pytest.fixture
def live_root(tmp_path: Path) -> Path:
    root = tmp_path / "clinicare"
    (root / "scripts").mkdir(parents=True)
    (root / CLINICARE_RUN_CASE).write_text("")
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    _tasks(root, TASK)
    return root


def _fake_run_case(
    monkeypatch,
    *,
    returncode: int = 0,
    trial: Optional[Path] = None,
    stderr: str = "",
    graded: Optional[bool] = None,
    grading: Optional[Dict[str, Any]] = None,
) -> List[List[str]]:
    calls: List[List[str]] = []

    def fake_run(cmd, timeout, **kwargs):
        calls.append(list(cmd))
        report: Dict[str, Any] = {"status": "ok", "trial_dir": str(trial)}
        if graded is not None:
            report["graded"] = graded
        if grading is not None:
            report["grading"] = grading
        stdout = json.dumps(report) + "\n" if trial else ""
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(clinicare_client, "require_docker", lambda: None)
    monkeypatch.setattr(clinicare_client, "run_in_process_group", fake_run)
    return calls


def _payload(result) -> Dict[str, Any]:
    return json.loads(result.completions[0].text)


def test_unmapped_model_fails_closed():
    envelope = {"clinicare_protocol": CLINICARE_PROTOCOL, "task_id": TASK, "evaluated_model": "openai/gpt-4o"}
    with pytest.raises(NonRetriableException, match="No CliniCARE system"):
        CliniCAREClient().make_request(Request(model="openai/gpt-4o", prompt=json.dumps(envelope)))


def test_job_name_is_deterministic_and_separates_builds(tmp_path):
    smoke, full = tmp_path / "tasks_smoke", tmp_path / "tasks"
    name = job_name("codex-gptx", TASK, full)
    assert name == job_name("codex-gptx", TASK, full)  # re-run resumes the same job
    assert name.startswith("medhelm_codex-gptx_") and name.endswith(f"_{TASK}")
    # Same task id in two builds must not share a job dir (harbor refuses to resume across them).
    assert name != job_name("codex-gptx", TASK, smoke)
    assert "_eff-high_" in job_name("codex-gptx", TASK, full, "high")


def test_read_trial_takes_task_id_from_task_path(tmp_path):
    long_id = "placeholder-scenario-with-a-long-name-000001"
    trial = _trial(tmp_path / "job", long_id)
    assert not trial.name.startswith(long_id)
    assert read_trial(trial)["task_id"] == long_id


@pytest.mark.parametrize(
    "verified,exc,status",
    [(True, None, "ok"), (True, "AgentTimeoutError", "agent_error"), (False, "RuntimeError", "dead")],
)
def test_read_trial_status(tmp_path, verified, exc, status):
    assert read_trial(_trial(tmp_path / "job", TASK, verified=verified, exc=exc))["status"] == status


def test_live_runs_run_case_and_returns_reward(monkeypatch, live_root, tmp_path):
    trial = _trial(tmp_path / "jobs" / "j", TASK)
    calls = _fake_run_case(monkeypatch, trial=trial)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))

    assert result.success
    assert _payload(result)["reward"]["score"] == 1.0
    cmd = calls[0]
    assert cmd[:2] == [str(live_root / ".venv" / "bin" / "python"), CLINICARE_RUN_CASE]
    assert cmd[cmd.index("--system") + 1] == "codex-gptx"
    assert cmd[cmd.index("--task-id") + 1] == TASK
    assert cmd[cmd.index("--job-name") + 1] == job_name("codex-gptx", TASK, live_root / "benchmark" / "tasks")
    assert "--grade" not in cmd


def test_live_passes_grade_and_effort(monkeypatch, live_root, tmp_path):
    calls = _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK))
    CliniCAREClient().make_request(_request(clinicare_root=str(live_root), grade=True, effort="high"))
    assert "--grade" in calls[0]
    assert calls[0][calls[0].index("--effort") + 1] == "high"


@pytest.mark.parametrize(
    "returncode,stderr",
    [
        (2, "run_case: OPENAI_API_KEY not set"),  # preflight refusal
        (2, "run_case: harbor refused the job (exit 1): FileExistsError"),  # stale job dir
        (1, "Traceback (most recent call last): ModuleNotFoundError"),  # run_case itself crashed
    ],
)
def test_any_run_case_failure_is_fatal(monkeypatch, live_root, returncode, stderr):
    # Per-case outcomes always come back as exit 0, so any other exit is a broken setup: abort the
    # run instead of recording an empty result for every remaining case.
    _fake_run_case(monkeypatch, returncode=returncode, stderr=stderr)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert not result.success
    assert result.error_flags is not None and result.error_flags.is_fatal and not result.error_flags.is_retriable


def test_timeout_kills_the_whole_process_group(tmp_path):
    # run_case.py starts harbor as a child; killing only run_case.py would leave harbor running.
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        clinicare_client.run_in_process_group([sys.executable, "-c", script], 2, stdout=subprocess.PIPE)
    grandchild = int(pid_file.read_text())
    for _ in range(50):  # the kill is asynchronous; give the OS a moment to reap it
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(grandchild, 9)
    pytest.fail("grandchild survived the timeout")


def test_live_dead_trial_is_non_fatal_and_not_retried(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK, verified=False, exc="RuntimeError"))
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert not result.success
    assert result.error_flags is not None
    assert not result.error_flags.is_fatal and not result.error_flags.is_retriable


def test_live_timeout_is_non_fatal(monkeypatch, live_root):
    def timeout(cmd, timeout, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(clinicare_client, "require_docker", lambda: None)
    monkeypatch.setattr(clinicare_client, "run_in_process_group", timeout)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert result.error_flags is not None and not result.error_flags.is_fatal


def test_live_misattributed_trial_raises(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK, model="openai/gpt-5.4"))
    with pytest.raises(NonRetriableException, match="refusing to attribute"):
        CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))


def test_live_missing_docker_is_fatal(monkeypatch, live_root):
    def no_docker():
        raise RuntimeError("no docker")

    monkeypatch.setattr(clinicare_client, "require_docker", no_docker)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert result.error_flags is not None and result.error_flags.is_fatal


def test_live_old_checkout_raises(tmp_path):
    (tmp_path / "old").mkdir()
    with pytest.raises(NonRetriableException, match="update the CliniCARE checkout"):
        CliniCAREClient().make_request(_request(clinicare_root=str(tmp_path / "old")))


def test_replay_serves_trial(tmp_path):
    tasks = _tasks(tmp_path, TASK, OTHER)
    _trial(tmp_path / "job", TASK)
    result = CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))
    assert result.success and _payload(result)["status"] == "ok"


def test_replay_missing_trial_is_non_fatal(tmp_path):
    tasks = _tasks(tmp_path, TASK, OTHER)
    _trial(tmp_path / "job", OTHER)
    result = CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))
    assert result.error_flags is not None and not result.error_flags.is_fatal and not result.error_flags.is_retriable


def test_replay_mixed_models_raise(tmp_path):
    tasks = _tasks(tmp_path, TASK, OTHER)
    _trial(tmp_path / "job", TASK)
    _trial(tmp_path / "job", OTHER, model="openai/gpt-5.4")
    with pytest.raises(NonRetriableException, match="mixes models"):
        CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))


def test_replay_misattributed_job_raises(tmp_path):
    tasks = _tasks(tmp_path, TASK)
    _trial(tmp_path / "job", TASK, model="openai/gpt-5.4")
    with pytest.raises(NonRetriableException, match="refusing to attribute"):
        CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))


@pytest.mark.parametrize("make_dir", [False, True])
def test_replay_missing_or_empty_job_dir_raises(tmp_path, make_dir):
    # A mistyped job_dir must not produce a "successful" run with no scores.
    tasks = _tasks(tmp_path, TASK)
    job = tmp_path / "job"
    if make_dir:
        job.mkdir()
    with pytest.raises(NonRetriableException, match="not found|no finished trials"):
        CliniCAREClient().make_request(_request(job_dir=str(job), tasks_dir=str(tasks)))


def test_replay_foreign_task_raises(tmp_path):
    tasks = _tasks(tmp_path, TASK)
    _trial(tmp_path / "job", TASK)
    _trial(tmp_path / "job", OTHER)
    with pytest.raises(NonRetriableException, match="different builds"):
        CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))


def test_task_checksum_matches_harbor(tmp_path):
    assert task_checksum(_tasks(tmp_path, TASK) / TASK) == TASK_CHECKSUM


@pytest.mark.parametrize("checksum", ["0" * 64, None])
def test_replay_stale_task_build_raises(tmp_path, checksum):
    # Same task ids, different contents (a CliniCARE update rebuilt in place): the old trials would
    # be scored against the new prompts.
    tasks = _tasks(tmp_path, TASK, OTHER)
    _trial(tmp_path / "job", TASK)
    _trial(tmp_path / "job", OTHER, checksum=checksum)
    with pytest.raises(NonRetriableException, match="different version"):
        CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))


@pytest.mark.parametrize(
    "reward,status",
    [
        # The verifier failed before scoring; its 0.0 is not a verdict (judge.py / test.sh paths).
        ({"score": 0.0, "judge_failed": 1, "rubrics_load_failed": 1}, "dead"),
        ({"score": 0.0, "judge_failed": 1, "judge_launch_failed": 1}, "dead"),
        # A missing report is a genuine 0, and judge_failed alone flags only the process pass.
        ({"score": 0.0, "report_present": 0, "report_chars": 0}, "ok"),
        ({"score": 1.0, "outcome_score": 1.0, "judge_failed": 1}, "ok"),
    ],
)
def test_verifier_failure_is_unscored_not_a_wrong_verdict(tmp_path, reward, status):
    trial = read_trial(_trial(tmp_path / "job", TASK, reward=reward))
    assert trial["status"] == status
    assert (trial["reward"] is None) == (status == "dead")


def test_replay_verifier_failure_is_a_coverage_gap(tmp_path):
    tasks = _tasks(tmp_path, TASK)
    _trial(tmp_path / "job", TASK, reward={"score": 0.0, "judge_failed": 1, "rubrics_load_failed": 1})
    result = CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))
    assert not result.success  # -> clinicare_trial_failed, not clinicare_score = 0
    assert result.error_flags is not None and not result.error_flags.is_fatal


GRADED_REWARD = {"score": 1.0, "process_pct": 75.0, "policy_support_judged": 1, "policy_citation_support_rate": 0.5}


def _warnings(monkeypatch) -> List[str]:
    seen: List[str] = []
    monkeypatch.setattr(clinicare_client, "hwarn", lambda msg, **kw: seen.append(msg))
    return seen


def _judge(outcome: str, reason: str, detail: Optional[str] = None) -> Dict[str, Any]:
    return {"outcome": outcome, "reason": reason, "detail": detail}


SCORED = _judge("scored", "scored")
NO_KEYS = _judge("failed", "judge_credentials_missing", "JUDGE_BASE_URL, JUDGE_API_KEY not set")


def _live(monkeypatch, live_root, tmp_path, *, reward=None, grading=None, graded=None, exc=None, grade=True):
    trial = _trial(tmp_path / "jobs" / "j", TASK, reward=reward, exc=exc)
    _fake_run_case(monkeypatch, trial=trial, grading=grading, graded=graded)
    warnings = _warnings(monkeypatch)
    knobs = {"clinicare_root": str(live_root), **({"grade": True} if grade else {})}
    payload = _payload(CliniCAREClient().make_request(_request(**knobs)))
    return payload, warnings, trial


def test_live_missing_judge_keys_warn_once_as_a_credentials_problem(monkeypatch, live_root, tmp_path):
    _fake_run_case(
        monkeypatch,
        trial=_trial(tmp_path / "jobs" / "j", TASK),
        graded=False,
        grading={"process": NO_KEYS, "policy": NO_KEYS},
    )
    warnings = _warnings(monkeypatch)
    client = CliniCAREClient()
    for _ in range(3):  # three ungraded cases in one run -> one warning, three counted in the stat
        payload = _payload(client.make_request(_request(clinicare_root=str(live_root), grade=True)))
        assert (payload["grade_requested"], payload["graded"]) == (True, False)
        assert payload["reward"]["score"] == 1.0  # the deterministic score is kept
    assert len(warnings) == 1
    assert (
        warnings[0].startswith("CliniCARE [credentials] judge credentials are missing")
        and "JUDGE_API_KEY" in warnings[0]
    )


@pytest.mark.parametrize(
    "process,tag,points_to",
    [
        (
            _judge("failed", "judge_credentials_rejected", "AuthenticationError: Error code: 401"),
            "[credentials] the judge API rejected the judge key",
            "JUDGE_API_KEY",
        ),
        (
            _judge("failed", "judge_error", "RateLimitError: Error code: 429"),
            "[judge] the process judge failed",
            "verifier/process.json",
        ),
        (_judge("failed", "no_rubrics"), "[task build] the process judge found no usable rubric bundle", "verifier/"),
        (
            _judge("failed", "judge_script_failed", "exit 1, no process_summary.json"),
            "[judge] the process judge did not finish",
            ".log",
        ),
    ],
)
def test_live_each_judge_failure_names_its_cause(monkeypatch, live_root, tmp_path, process, tag, points_to):
    payload, warnings, _ = _live(
        monkeypatch, live_root, tmp_path, graded=False, grading={"process": process, "policy": SCORED}
    )
    assert payload["graded"] is False
    assert len(warnings) == 1 and warnings[0].startswith(f"CliniCARE {tag}") and points_to in warnings[0]


@pytest.mark.parametrize("policy", [_judge("skipped", "policy_off"), _judge("skipped", "no_policy_rubrics")])
def test_live_by_design_skips_are_graded_and_silent(monkeypatch, live_root, tmp_path, policy):
    payload, warnings, _ = _live(
        monkeypatch, live_root, tmp_path, graded=True, grading={"process": SCORED, "policy": policy}
    )
    assert payload["graded"] is True and warnings == []


def test_live_no_report_is_an_agent_problem_not_a_grading_one(monkeypatch, live_root, tmp_path):
    no_report = _judge("skipped", "no_report")
    payload, warnings, trial = _live(
        monkeypatch,
        live_root,
        tmp_path,
        reward={"score": 0.0, "report_present": 0},
        graded=True,
        grading={"process": no_report, "policy": no_report},
    )
    assert payload["graded"] is True
    assert len(warnings) == 1 and warnings[0].startswith("CliniCARE [agent] the agent finished without writing")
    assert f"{trial}/agent/" in warnings[0]


def test_live_previously_graded_case_counts_as_graded(monkeypatch, live_root, tmp_path):
    # This run's judges failed (the keys are now missing), but an earlier run already wrote both
    # judges' scores: the judged columns are filled, so no warning.
    payload, warnings, _ = _live(
        monkeypatch,
        live_root,
        tmp_path,
        reward=GRADED_REWARD,
        graded=False,
        grading={"process": NO_KEYS, "policy": NO_KEYS},
    )
    assert payload["graded"] is True and warnings == []


@pytest.mark.parametrize("reward,graded_flag,graded", [(GRADED_REWARD, None, True), (None, False, False)])
def test_live_older_run_case_without_grading_report(monkeypatch, live_root, tmp_path, reward, graded_flag, graded):
    payload, warnings, _ = _live(monkeypatch, live_root, tmp_path, reward=reward, graded=graded_flag)
    assert payload["graded"] is graded
    assert (warnings == []) is graded
    if not graded:
        assert "does not report why" in warnings[0]


def test_live_without_grade_never_warns_about_grading(monkeypatch, live_root, tmp_path):
    payload, warnings, _ = _live(monkeypatch, live_root, tmp_path, graded=False, grade=False)
    assert payload["grade_requested"] is False and warnings == []


@pytest.mark.parametrize(
    "exc,reward,tag,points_to",
    [
        ("NonZeroAgentExitCodeError", None, "[agent] the agent raised NonZeroAgentExitCodeError: boom", "/agent/"),
        (
            None,
            {"score": 0.0, "evidence_calls_auth_error": 4},
            "[credentials] 4 MIMIC tool call(s) were rejected",
            "MIMIC_API_SECRET",
        ),
        (None, {"score": 1.0, "environment_degraded": 1}, "[environment] the agent reported a degraded", "report.md"),
    ],
)
def test_scored_trial_problems_name_their_cause(monkeypatch, live_root, tmp_path, exc, reward, tag, points_to):
    _, warnings, _ = _live(monkeypatch, live_root, tmp_path, reward=reward, exc=exc, grade=False)
    assert len(warnings) == 1 and warnings[0].startswith(f"CliniCARE {tag}") and points_to in warnings[0]


@pytest.mark.parametrize(
    "verified,reward,tag",
    [
        (
            True,
            {"score": 0.0, "judge_failed": 1, "rubrics_load_failed": 1},
            "[task build] the verifier could not load this task's rubric bundle",
        ),
        (
            True,
            {"score": 0.0, "judge_failed": 1, "judge_launch_failed": 1},
            "[environment] the verifier could not start its judge",
        ),
        (False, None, "[environment] harbor recorded no verifier result (RuntimeError: boom)"),
    ],
)
def test_dead_trial_error_names_the_part_that_failed(monkeypatch, live_root, tmp_path, verified, reward, tag):
    trial = _trial(
        tmp_path / "jobs" / "j", TASK, verified=verified, reward=reward, exc=None if verified else "RuntimeError"
    )
    _fake_run_case(monkeypatch, trial=trial)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert not result.success and result.error is not None
    assert result.error.startswith(f"CliniCARE {tag}") and str(trial) in result.error


def test_timeout_error_names_the_job_log(monkeypatch, live_root):
    def timeout(cmd, timeout, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(clinicare_client, "require_docker", lambda: None)
    monkeypatch.setattr(clinicare_client, "run_in_process_group", timeout)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root), case_timeout_sec=60))
    assert result.error is not None and result.error.startswith("CliniCARE [timeout]")
    assert (
        "case_timeout_sec=60s" in result.error
        and job_name("codex-gptx", TASK, live_root / "benchmark" / "tasks") + ".log" in result.error
    )


def _replay(tmp_path, monkeypatch, *, reward=None, summaries=(), record=None):
    tasks = _tasks(tmp_path, TASK)
    trial = _trial(tmp_path / "job", TASK, reward=reward)
    for name in summaries:
        (tmp_path / "job" / name).write_text("{}")
    if record is not None:
        (trial / "verifier" / "process.json").write_text(json.dumps(record))
    warnings = _warnings(monkeypatch)
    result = CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks), grade=True))
    return _payload(result), warnings


BOTH_SUMMARIES = ("process_summary.json", "policy_support_summary.json")


def test_replay_graded_job_is_graded(monkeypatch, tmp_path):
    payload, warnings = _replay(tmp_path, monkeypatch, reward=GRADED_REWARD, summaries=BOTH_SUMMARIES)
    assert payload["graded"] is True and warnings == []


def test_replay_never_graded_job_says_so(monkeypatch, tmp_path):
    payload, warnings = _replay(tmp_path, monkeypatch, reward={"score": 1.0})
    assert payload["graded"] is False
    assert len(warnings) == 1 and warnings[0].startswith("CliniCARE [judge] this replayed job was never graded")


def test_replay_rejected_judge_key_is_a_credentials_problem(monkeypatch, tmp_path):
    payload, warnings = _replay(
        tmp_path,
        monkeypatch,
        reward={"score": 1.0, "policy_support_judged": 1},
        summaries=BOTH_SUMMARIES,
        record={"error": "AuthenticationError: Error code: 401 - invalid key"},
    )
    assert payload["graded"] is False
    assert len(warnings) == 1 and warnings[0].startswith("CliniCARE [credentials] the judge API rejected")


def test_replay_nothing_to_score_is_graded(monkeypatch, tmp_path):
    # The judges ran over the job and left no record for this trial (e.g. a policy-off task).
    payload, warnings = _replay(
        tmp_path, monkeypatch, reward={"score": 1.0, "process_pct": 50.0}, summaries=BOTH_SUMMARIES
    )
    assert payload["graded"] is True and warnings == []


def test_different_problems_each_warn_once(monkeypatch, live_root, tmp_path):
    trial = _trial(
        tmp_path / "jobs" / "j", TASK, reward={"score": 0.0, "report_present": 0, "evidence_calls_auth_error": 2}
    )
    _fake_run_case(monkeypatch, trial=trial, graded=False, grading={"process": NO_KEYS, "policy": NO_KEYS})
    warnings = _warnings(monkeypatch)
    client = CliniCAREClient()
    for _ in range(3):
        client.make_request(_request(clinicare_root=str(live_root), grade=True))
    tags = sorted(w.split("]")[0] + "]" for w in warnings)
    assert tags == ["CliniCARE [agent]", "CliniCARE [credentials]", "CliniCARE [credentials]"]


def test_each_run_logs_its_own_warnings(monkeypatch, live_root, tmp_path):
    # One medhelm-run process can run several run specs; a problem warned in one must still be
    # warned in the next.
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK), graded=False)
    warnings = _warnings(monkeypatch)
    client = CliniCAREClient()
    for run in ({}, {"effort": "high"}):
        for _ in range(2):
            client.make_request(_request(clinicare_root=str(live_root), grade=True, **run))
    assert len(warnings) == 2


def test_warning_and_docker_check_are_shared_across_client_instances(monkeypatch, live_root, tmp_path):
    # With --num-threads N, HELM's AutoClient can build N instances of this client concurrently. A
    # real 5-thread replay logged the "not graded" warning 5 times before this was shared.
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK), graded=False)
    docker_checks: List[int] = []
    monkeypatch.setattr(clinicare_client, "require_docker", lambda: docker_checks.append(1))
    warnings = _warnings(monkeypatch)
    for _ in range(5):
        CliniCAREClient().make_request(_request(clinicare_root=str(live_root), grade=True))
    assert len(warnings) == 1
    assert len(docker_checks) == 1
