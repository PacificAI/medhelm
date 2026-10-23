from helm.benchmark.scenarios.medgrounding.pg26_base_scenarios import PG26BaseScenario


class PG26AblationScenario(PG26BaseScenario):
    """PacificMedGrounding PG26 Ablation: a necessary fact removed, so the key changes."""

    name = "pg26_ablation"
    display_name = "Ablation"
    description = (
        "Evaluates whether a model moves its answer when a necessary clinical fact is removed from the record, "
        "choosing the ascertainment or revised action instead of the original one."
    )
    what = "Assess sensitivity of clinical management decisions to removal of a necessary fact"


class PG26AbstentionScenario(PG26BaseScenario):
    """PacificMedGrounding PG26 Abstention: the record cannot decide the management."""

    name = "pg26_abstention"
    display_name = "Abstention"
    description = (
        "Evaluates whether a model correctly abstains when the record cannot decide the management, and does not "
        "commit to a treatment."
    )
    what = "Assess calibrated abstention in clinical management decisions"


class PG26ContradictionScenario(PG26BaseScenario):
    """PacificMedGrounding PG26 Contradiction: two sources disagree about a decisive fact."""

    name = "pg26_contradiction"
    display_name = "Contradiction"
    description = (
        "Evaluates whether a model reconciles or re-measures when two dated sources in the record disagree, rather "
        "than acting on either value."
    )
    what = "Assess sensitivity of clinical management decisions to conflicting sources"


class PG26InversionScenario(PG26BaseScenario):
    """PacificMedGrounding PG26 Inversion: one finding replaced by its opposite, so the key flips."""

    name = "pg26_inversion"
    display_name = "Inversion"
    description = (
        "Evaluates whether a model flips its answer when a decisive finding is replaced by its clinical opposite, "
        "keeping the full-record answer is a failure."
    )
    what = "Assess sensitivity of clinical management decisions to inverted decisive findings"
    sheet_name = "Test cases"
    layout = "paired"


class PG26MisfiledScenario(PG26BaseScenario):
    """PacificMedGrounding PG26 Misfiled: a result that cannot belong to this patient."""

    name = "pg26_misfiled"
    display_name = "Misfiled"
    description = (
        "Evaluates whether a model queries the record when a result is implausible for the patient, rather than "
        "acting on the misfiled result."
    )
    what = "Assess sensitivity of clinical management decisions to misfiled results"


class PG26StalenessScenario(PG26BaseScenario):
    """PacificMedGrounding PG26 Staleness: the decisive value is present but outdated."""

    name = "pg26_staleness"
    display_name = "Staleness"
    description = (
        "Evaluates whether a model re-measures before acting when the decisive value is present but outdated, rather "
        "than acting on the stale value."
    )
    what = "Assess sensitivity of clinical management decisions to outdated decisive values"
