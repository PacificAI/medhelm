"""Shared constants for the CliniCARE-Bench MedHELM integration."""

# Where the CliniCARE checkout lives (live mode runs cases through it) and, optionally, which built
# tasks dir to use (default: <CLINICARE_ROOT>/benchmark/tasks).
CLINICARE_ROOT_ENV = "CLINICARE_ROOT"
CLINICARE_TASKS_DIR_ENV = "CLINICARE_TASKS_DIR"
# The CliniCARE entry point that runs one case and prints the trial as JSON.
CLINICARE_RUN_CASE = "scripts/run_case.py"

# reward.json markers the in-container verifier writes when it failed before scoring the outcome
# (judge.py: rubric bundle failed to load; test.sh: judge.py failed to launch). The reward still
# says score 0.0, which is not a verdict, so the trial is unscored. Mirrors CliniCARE's
# clinicare_core.rewards.VERIFIER_FAILED_KEYS (MedHELM imports no CliniCARE code). Not judge_failed:
# that flags only the secondary process pass.
VERIFIER_FAILED_KEYS = ("rubrics_load_failed", "judge_launch_failed")

# What the openai SDK raises when CliniCARE's judge API refuses the key (401 / 403); mirrors
# run_case.py. The judges record an error as "<ExceptionType>: <message>", so only the type is matched.
REJECTED_KEY_ERRORS = ("AuthenticationError", "PermissionDeniedError")

# Gold verdicts as they appear in the cohort bundle's expected_label column.
VERDICT_LABELS = (
    "YES",
    "NO",
    "INDETERMINATE_LACK_OF_DATA",
    "INDETERMINATE_MEDICAL_AMBIGUITY",
)

CLINICARE_PROTOCOL = "clinicare.v1"
# Internal routing deployment. Leaderboard rows use the evaluated model, never this name.
CLINICARE_HARNESS_DEPLOYMENT = "clinicare/harness"
