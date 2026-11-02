"""Runs (or replays) CliniCARE-Bench agent episodes for MedHELM.

Live (default): each request runs one case through the CliniCARE checkout's ``scripts/run_case.py``
in CliniCARE's own virtualenv, then reads the trial it produced. MedHELM never imports CliniCARE
code or its harbor pin; the subprocess command and its JSON output are the whole contract.

Replay (``job_dir=`` on the run entry): reads a finished CliniCARE job dir instead. No Docker, MIMIC
or model calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from helm.benchmark.scenarios.clinicare_constants import (
    CLINICARE_PROTOCOL,
    CLINICARE_RUN_CASE,
    REJECTED_KEY_ERRORS,
    VERIFIER_FAILED_KEYS,
)
from helm.benchmark.scenarios.clinicare_scenario import resolve_clinicare_root, resolve_tasks_dir
from helm.clients.client import Client
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.request import ErrorFlags, GeneratedOutput, Request, RequestResult
from helm.proxy.retry import NonRetriableException

# run_case.py reports every per-case outcome (ok / agent_error / dead) with exit 0. Any other exit
# means the setup or the job is broken (missing key, unbuilt tasks, gateway not allowlisted, harbor
# refusing a job, a stale trial, a crash), so every remaining case would fail the same way: abort.
# CliniCARE's agent ceiling is 5400 s, plus up to 3x the 360 s harness setup and the verifier.
DEFAULT_CASE_TIMEOUT_SEC = 7200


def load_model_map() -> Dict[str, Any]:
    map_path = Path(__file__).resolve().parents[1] / "benchmark" / "static" / "clinicare_model_map.yaml"
    with map_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def lookup_system(evaluated_model: str) -> Dict[str, Any]:
    """The model-map row for ``evaluated_model``. Exact match only; an unmapped model fails closed."""
    for row in load_model_map().get("models") or []:
        if row.get("model") == evaluated_model:
            return row
    raise NonRetriableException(
        f"No CliniCARE system for model {evaluated_model!r}; add it to benchmark/static/clinicare_model_map.yaml"
    )


def require_docker() -> None:
    """Fail closed if the Docker daemon is not running: every case runs in containers."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=20)
    except FileNotFoundError as exc:
        raise RuntimeError("CliniCARE needs Docker to run cases, but the docker CLI was not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("CliniCARE Docker preflight timed out (`docker info`). Is Docker running?") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or f"exit {result.returncode}"
        raise RuntimeError(f"CliniCARE needs a running Docker daemon. `docker info` failed: {detail}")


def job_name(system: str, task_id: str, tasks_dir: Path, effort: Optional[str] = None) -> str:
    """One harbor job per (system, effort, tasks build, case). Deterministic, so re-running resumes it.

    The tasks-dir hash keeps two builds that share task ids (a smoke dir and the full build) from
    colliding on one job dir, which harbor would refuse to resume.
    """
    build = hashlib.sha256(str(Path(tasks_dir).resolve()).encode()).hexdigest()[:8]
    return f"medhelm_{system}" + (f"_eff-{effort}" if effort else "") + f"_{build}_{task_id}"


def run_in_process_group(cmd: list, timeout: int, **kwargs: Any) -> subprocess.CompletedProcess:
    """subprocess.run, but a timeout kills the whole process group: run_case.py's harbor child and
    its containers' clients, not just run_case.py (which would leave harbor running and writing)."""
    with subprocess.Popen(cmd, start_new_session=True, **kwargs) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def task_checksum(task_dir: Path) -> str:
    """harbor's task checksum (``Task.checksum``, i.e. ``dirhash(task_dir, "sha256")`` with dirhash's
    defaults), which harbor records as ``task_checksum`` in each trial's result.json. Reimplemented so
    MedHELM needs no dirhash dependency; symlinks are followed and empty dirs are skipped, as there."""

    def walk(directory: Path) -> Optional[str]:
        entries = []
        for path in directory.iterdir():
            if path.is_dir():
                digest = walk(path)
                if digest is not None:
                    entries.append(f"dirhash:{digest}\000name:{path.name}")
            else:
                data = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(2**20), b""):
                        data.update(chunk)
                entries.append(f"data:{data.hexdigest()}\000name:{path.name}")
        if not entries:
            return None
        return hashlib.sha256("\000\000".join(sorted(entries)).encode("utf-8")).hexdigest()

    digest = walk(task_dir)
    if digest is None:
        raise ValueError(f"{task_dir}: nothing to hash")
    return digest


def read_trial(trial_dir: Path) -> Dict[str, Any]:
    """Status, rewards and identity of one finished trial (the run_case.py status rules)."""
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    reward_path = trial_dir / "verifier" / "reward.json"
    verified = bool((result.get("verifier_result") or {}).get("rewards")) and reward_path.is_file()
    reward = json.loads(reward_path.read_text(encoding="utf-8")) if verified else None
    verifier_failure = next((k for k in VERIFIER_FAILED_KEYS if (reward or {}).get(k) == 1), None)
    if verifier_failure:
        verified, reward = False, None  # the verifier failed before scoring: its 0.0 is not a verdict
    exception = result.get("exception_info") or {}
    report = trial_dir / "verifier" / "report.md"
    agent = (result.get("config") or {}).get("agent") or {}
    return {
        "status": "dead" if not verified else ("agent_error" if exception else "ok"),
        # harbor truncates trial dir names to 32 chars, so the task id comes from the task path.
        "task_id": Path(((result.get("task_id") or {}).get("path")) or "").name,
        "task_checksum": result.get("task_checksum"),
        "model_name": agent.get("model_name"),
        "agent_version": (result.get("agent_info") or {}).get("version"),
        "exception_type": exception.get("exception_type"),
        "exception_message": (exception.get("exception_message") or "")[:300],
        "verifier_failure": verifier_failure,
        "reward": reward,
        "report_chars": len(report.read_text(encoding="utf-8")) if report.is_file() else 0,
    }


# -- diagnosis --------------------------------------------------------------------------------------
# Every warning and per-case error starts with a category, so the reader knows which debugging path
# to take before reading further: [credentials] a key is missing or was rejected; [agent] the model's
# own behaviour (it raised, or wrote no report); [judge] an LLM judge failed; [task build] the built
# task is incomplete; [environment] the container, harness or tools failed; [timeout] the case ran
# out of time. Each names what to check and the file to open.

# Per-trial record each judge writes (run_case.py's JUDGES); judge -> (file, job summary).
JUDGE_FILES = {
    "process": ("process.json", "process_summary.json"),
    "policy": ("policy_support.json", "policy_support_summary.json"),
}


def dead_trial_message(trial: Dict[str, Any], trial_dir: Path) -> str:
    """Why a trial has no score, by which part failed."""
    if trial["verifier_failure"] == "rubrics_load_failed":
        return (
            "[task build] the verifier could not load this task's rubric bundle, so the case is unscored. "
            f"Check the task build; see {trial_dir}/verifier/test-stdout.txt"
        )
    if trial["verifier_failure"] == "judge_launch_failed":
        return (
            "[environment] the verifier could not start its judge inside the container, so the case is "
            f"unscored. See {trial_dir}/verifier/test-stdout.txt"
        )
    exc = f" ({trial['exception_type']}: {trial['exception_message']})" if trial["exception_type"] else ""
    return (
        f"[environment] harbor recorded no verifier result{exc}: the container or harness failed before "
        f"scoring. See {trial_dir}/trial.log and exception_info in {trial_dir}/result.json"
    )


def trial_issues(trial: Dict[str, Any], trial_dir: Path) -> list:
    """(warning key, message) for each problem in a scored trial."""
    reward = trial["reward"] or {}
    issues = []
    if trial["status"] == "agent_error":
        issues.append(
            (
                "agent_error",
                f"[agent] the agent raised {trial['exception_type']}: {trial['exception_message']}. The case is "
                f"still scored and counts in clinicare_trial_failed. Debug the agent: {trial_dir}/agent/ and "
                f"{trial_dir}/trial.log",
            )
        )
    if reward.get("report_present") == 0:
        issues.append(
            (
                "no_report",
                "[agent] the agent finished without writing a report, so the case scores 0 and the judges have "
                f"nothing to grade. This is the agent's behaviour, not a grading problem. See {trial_dir}/agent/",
            )
        )
    if (reward.get("evidence_calls_auth_error") or 0) > 0:
        issues.append(
            (
                "mimic_auth",
                f"[credentials] {int(reward['evidence_calls_auth_error'])} MIMIC tool call(s) were rejected for "
                "credentials, so the agent may have worked without patient data and its score may not reflect "
                f"the model. Check MIMIC_API_SECRET in CliniCARE's .env; see {trial_dir}/verifier/evidence.jsonl",
            )
        )
    if reward.get("environment_degraded") == 1:
        issues.append(
            (
                "environment_degraded",
                "[environment] the agent reported a degraded environment (its tools were failing). Check the "
                f"MIMIC tool sidecar; see the agent's note in {trial_dir}/verifier/report.md",
            )
        )
    return issues


def scored_in_reward(reward: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Which judges' scores this reward.json already carries (from this run or an earlier one)."""
    reward = reward or {}
    return {"process": "process_pct" in reward, "policy": reward.get("policy_support_judged") == 1}


def live_grading(report: Dict[str, Any], reward: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """run_case.py's ``grading`` report, except that a judge whose score an earlier run already wrote
    counts as scored even if it failed this time (e.g. the judge keys were since removed)."""
    if "grading" not in report:  # an older CliniCARE checkout: it reports only the overall flag
        if report.get("graded") or all(scored_in_reward(reward).values()):
            return {}
        return {"judges": {"outcome": "failed", "reason": "unreported", "detail": None}}
    grading = dict(report.get("grading") or {})
    for judge, already in scored_in_reward(reward).items():
        if already and (grading.get(judge) or {}).get("outcome") == "failed":
            grading[judge] = {"outcome": "scored", "reason": "scored_earlier", "detail": None}
    return grading


def _judge_failure(error: str) -> Dict[str, Any]:
    kind = error.split(":", 1)[0].strip()
    reason = "judge_credentials_rejected" if kind in REJECTED_KEY_ERRORS else "judge_error"
    return {"outcome": "failed", "reason": reason, "detail": error}


def replay_grading(trial: Dict[str, Any], trial_dir: Path) -> Dict[str, Dict[str, Any]]:
    """run_case.py's ``grading`` report, rebuilt from the files a CliniCARE grading run left behind."""
    reward = trial["reward"] or {}
    scored = scored_in_reward(reward)
    grading = {}
    for judge, (trial_file, summary_file) in JUDGE_FILES.items():
        try:
            error = json.loads((trial_dir / "verifier" / trial_file).read_text(encoding="utf-8")).get("error")
        except (OSError, ValueError):
            error = None
        if scored[judge]:
            grading[judge] = {"outcome": "scored", "reason": "scored", "detail": None}
        elif error:
            grading[judge] = _judge_failure(error)
        elif reward.get("report_present") == 0:
            grading[judge] = {"outcome": "skipped", "reason": "no_report", "detail": None}
        elif not (trial_dir.parent / summary_file).is_file():
            grading[judge] = {"outcome": "failed", "reason": "not_graded", "detail": None}
        else:  # the judge ran over this job and left no record for the trial: nothing to score
            grading[judge] = {"outcome": "skipped", "reason": "nothing_to_score", "detail": None}
    return grading


def grading_issues(grading: Dict[str, Dict[str, Any]], trial_dir: Path, job_log: Optional[Path]) -> list:
    """(warning key, message) for each judge that failed. By-design skips (no report, policy-off,
    no rubrics of that kind) are not problems; a missing report is reported as an agent issue."""
    log = f"the case's job log {job_log}" if job_log else "the CliniCARE grading output"
    issues = []
    for judge, result in grading.items():
        reason, detail = result.get("reason"), result.get("detail")
        if result.get("outcome") != "failed":
            continue
        if reason == "judge_credentials_missing":
            message = (
                f"[credentials] judge credentials are missing ({detail}), so no case can be graded and "
                "clinicare_process_score / clinicare_policy_support will be blank. Set them in CliniCARE's .env"
            )
        elif reason == "judge_credentials_rejected":
            message = (
                f"[credentials] the judge API rejected the judge key ({detail}). Check JUDGE_API_KEY and "
                f"JUDGE_BASE_URL in CliniCARE's .env; first case: {trial_dir}"
            )
        elif reason == "judge_error":
            message = (
                f"[judge] the {judge} judge failed on a case ({detail}), usually a rate limit, timeout or "
                "malformed reply. Re-run the suite to retry; finished cases are reused. See "
                f"{trial_dir}/verifier/{JUDGE_FILES[judge][0]}"
            )
        elif reason == "no_rubrics":
            message = (
                f"[task build] the {judge} judge found no usable rubric bundle for a case, so it could not "
                f"score it. Check the task build; see {trial_dir}/verifier/"
            )
        elif reason == "not_graded":
            message = (
                "[judge] this replayed job was never graded in CliniCARE, so the judged columns are blank. "
                "Grade it first: benchmark/eval/metrics/{process,policy_support}.py <job dir> --amend-reward"
            )
        elif reason == "unreported":
            message = (
                "[judge] grading did not complete, and this CliniCARE checkout does not report why. Update "
                f"the CliniCARE checkout for a specific cause, or see {log}"
            )
        else:  # judge_script_failed, or a reason from a newer CliniCARE
            message = f"[judge] the {judge} judge did not finish ({reason}: {detail}). See {log}"
        issues.append((f"{judge}:{reason}" if reason in ("judge_error", "judge_script_failed") else reason, message))
    return issues


def _completion(trial: Dict[str, Any], system: str, grade_requested: bool, graded: bool) -> RequestResult:
    payload = {key: trial[key] for key in ("status", "reward", "report_chars", "agent_version", "exception_type")}
    payload.update(system=system, grade_requested=grade_requested, graded=graded)
    return RequestResult(
        success=True,
        cached=False,
        completions=[GeneratedOutput(text=json.dumps(payload), logprob=0.0, tokens=[])],
        embedding=[],
    )


def _failed(error: str, fatal: bool) -> RequestResult:
    # Non-fatal: MedHELM records an empty completion and keeps going. Never retriable: a dead trial
    # or a broken setup will not fix itself across HELM's 5 backoff retries.
    return RequestResult(
        success=False,
        cached=False,
        completions=[],
        embedding=[],
        error=error,
        error_flags=ErrorFlags(is_retriable=False, is_fatal=fatal),
    )


class CliniCAREClient(Client):
    """Serves each task's trial as the completion: JSON ``{"status", "reward", "report_chars", ...}``.

    Refuses (NonRetriableException) an unmapped model, a trial produced by a different model than
    the map declares, a replay job dir that mixes models, or one holding tasks absent from the tasks
    dir or built from a different version of them. Does not cache: the harbor job dir is the cache (a finished case is never re-run).

    Run-wide state is shared by every instance: with --num-threads N, HELM's AutoClient can build N
    instances of this client at once (its client cache is not populated under a lock), and the
    replay index, the Docker check and the once-per-run warning must still happen once.
    """

    _lock = threading.Lock()
    _replay_index: Dict[str, Dict[str, Path]] = {}
    _docker_checked = False
    _warned: set = set()  # (run, kind) pairs; a run is the envelope minus the task id

    def __init__(self, cache_config: Optional[CacheConfig] = None, **kwargs: Any):
        del cache_config, kwargs

    def _warn(self, envelope: Dict[str, Any], issues: list) -> None:
        """Log each kind of problem once per run, naming the first case it was seen in. The counts are
        in stats.json (clinicare_trial_failed, clinicare_grading_failed and the trust signals).

        One medhelm-run process can run several run specs, and the class state outlives each, so a
        warning is keyed by its run: every request of a run spec carries the same envelope but for the
        task id."""
        run = json.dumps({k: v for k, v in envelope.items() if k != "task_id"}, sort_keys=True)
        for key, message in issues:
            with self._lock:
                if (run, key) in self._warned:
                    continue
                self._warned.add((run, key))
            hwarn(f"CliniCARE {message} (logged once per run; see stats.json for how many cases)")

    def make_request(self, request: Request) -> RequestResult:
        envelope = json.loads(request.prompt)
        if envelope.get("clinicare_protocol") != CLINICARE_PROTOCOL:
            raise NonRetriableException(f"Expected protocol {CLINICARE_PROTOCOL}, got {envelope!r:.200}")
        row = lookup_system(envelope["evaluated_model"])
        if envelope.get("job_dir"):
            return self._replay(envelope, row)
        return self._live(envelope, row)

    # -- live -------------------------------------------------------------------------------------

    def _live(self, envelope: Dict[str, Any], row: Dict[str, Any]) -> RequestResult:
        root = resolve_clinicare_root(envelope.get("clinicare_root", ""))
        if root is None:
            raise NonRetriableException(
                "CliniCARE live mode needs the CliniCARE checkout: set clinicare_root= or export CLINICARE_ROOT"
            )
        python = root / ".venv" / "bin" / "python"
        if not (root / CLINICARE_RUN_CASE).is_file():
            raise NonRetriableException(f"{root} has no {CLINICARE_RUN_CASE}; update the CliniCARE checkout")
        if not python.is_file():
            raise NonRetriableException(f"No CliniCARE virtualenv at {python}; run `uv sync` in {root}")
        with self._lock:
            if not self._docker_checked:
                try:
                    require_docker()
                except RuntimeError as exc:
                    return _failed(str(exc), fatal=True)
                type(self)._docker_checked = True

        task_id = envelope["task_id"]
        system = row["system"]
        effort = envelope.get("effort") or row.get("effort")
        tasks_dir = resolve_tasks_dir(envelope.get("tasks_dir", ""), str(root))
        jobs_dir = Path(envelope.get("jobs_dir") or root / "jobs" / "medhelm").expanduser()
        cmd = [
            str(python),
            CLINICARE_RUN_CASE,
            "--system",
            system,
            "--task-id",
            task_id,
            "--tasks",
            str(tasks_dir),
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            job_name(system, task_id, tasks_dir, effort),
        ]
        if effort:
            cmd += ["--effort", effort]
        if envelope.get("grade"):
            cmd += ["--grade"]
        timeout = int(envelope.get("case_timeout_sec") or DEFAULT_CASE_TIMEOUT_SEC)
        try:
            proc = run_in_process_group(
                cmd,
                timeout,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return _failed(
                f"CliniCARE [timeout] the case ran past case_timeout_sec={timeout}s and was stopped. Check the "
                f"job log {jobs_dir / (cmd[cmd.index('--job-name') + 1] + '.log')}; raise case_timeout_sec if "
                "the agent needs longer",
                fatal=False,
            )
        if proc.returncode != 0:
            # Abort the run rather than record hundreds of empty results (see the note at the top).
            return _failed(
                f"CliniCARE run_case failed (exit {proc.returncode}): {proc.stderr.strip()[-2000:]}", fatal=True
            )
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        trial_dir = Path(report["trial_dir"])
        trial = read_trial(trial_dir)
        self._check_attribution(trial["model_name"], row, trial_dir)
        if trial["status"] == "dead":
            return _failed(f"CliniCARE {dead_trial_message(trial, trial_dir)}", fatal=False)
        self._warn(envelope, trial_issues(trial, trial_dir))
        grade_requested = bool(envelope.get("grade"))
        grading = live_grading(report, trial["reward"]) if grade_requested else {}
        graded = grade_requested and all(result.get("outcome") != "failed" for result in grading.values())
        if grade_requested:
            self._warn(envelope, grading_issues(grading, trial_dir, jobs_dir / f"{trial_dir.parent.name}.log"))
        return _completion(trial, system, grade_requested, graded)

    # -- replay -----------------------------------------------------------------------------------

    def _index_job(self, job_dir: str, tasks_dir: Path, row: Dict[str, Any]) -> Dict[str, Path]:
        job = Path(job_dir).expanduser()
        if not job.is_dir():
            raise NonRetriableException(f"CliniCARE job dir not found: {job}")
        trials: Dict[str, Path] = {}
        models = set()
        for trial in sorted(job.glob("*__*/")):
            info = read_trial(trial)
            if info["task_id"] in trials:
                raise NonRetriableException(f"Job dir {job} has more than one trial for one task")
            trials[info["task_id"]] = trial
            models.add(info["model_name"])
        if not trials:
            # Otherwise a wrong job_dir yields a "successful" run with no scores at all.
            raise NonRetriableException(f"CliniCARE job dir {job} has no finished trials")
        if len(models) > 1:
            raise NonRetriableException(f"Job dir {job} mixes models {sorted(models)}; one job dir per model")
        for model in models:
            self._check_attribution(model, row, job)
        foreign = [task_id for task_id in trials if not (tasks_dir / task_id).is_dir()]
        if foreign:
            raise NonRetriableException(
                f"{len(foreign)} trial(s) in {job} have no task in {tasks_dir}; the two are from different builds"
            )
        # The same task ids can survive a rebuild with different contents (a CliniCARE update), and the
        # old trials would then be scored as answers to the new prompts. Same check as run_case.py.
        stale = [
            task_id
            for task_id, trial in trials.items()
            if read_trial(trial)["task_checksum"] != task_checksum(tasks_dir / task_id)
        ]
        if stale:
            raise NonRetriableException(
                f"{len(stale)} trial(s) in {job} were built from a different version of their task than "
                f"{tasks_dir} (task checksum differs); replay against the tasks build that job ran on"
            )
        n_tasks = sum(1 for d in tasks_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))
        hlog(f"CliniCARE replay: {len(trials)} trials for {n_tasks} tasks ({job.name})")
        return trials

    def _replay(self, envelope: Dict[str, Any], row: Dict[str, Any]) -> RequestResult:
        key = envelope["job_dir"]
        tasks_dir = resolve_tasks_dir(envelope.get("tasks_dir", ""), envelope.get("clinicare_root", ""))
        with self._lock:
            if key not in self._replay_index:
                self._replay_index[key] = self._index_job(key, tasks_dir, row)
        trial_dir = self._replay_index[key].get(envelope["task_id"])
        if trial_dir is None:
            return _failed(f"no finished trial for task {envelope['task_id']}", fatal=False)
        trial = read_trial(trial_dir)
        if trial["status"] == "dead":
            return _failed(f"CliniCARE {dead_trial_message(trial, trial_dir)}", fatal=False)
        self._warn(envelope, trial_issues(trial, trial_dir))
        grade_requested = bool(envelope.get("grade"))
        grading = replay_grading(trial, trial_dir)
        graded = grade_requested and all(result["outcome"] != "failed" for result in grading.values())
        if grade_requested:
            self._warn(envelope, grading_issues(grading, trial_dir, None))
        return _completion(trial, row["system"], grade_requested, graded)

    @staticmethod
    def _check_attribution(model_name: Optional[str], row: Dict[str, Any], where: Path) -> None:
        if model_name != row["harbor_model"]:
            raise NonRetriableException(
                f"{where} was produced by {model_name!r}, but the model map says {row['model']!r} runs as "
                f"{row['harbor_model']!r}; refusing to attribute these results"
            )
