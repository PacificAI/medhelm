from helm.benchmark.scenarios.medrobustness.pr26_base_scenario import PR26BaseScenario


class PR26FormatScenario(PR26BaseScenario):
    """PacificMedRobustness PR26 Format: prose versus tabular record rendering."""

    name = "pr26_format"
    display_name = "Format"
    description = (
        "Evaluates whether clinically equivalent prose and tabular records produce the same correct management "
        "decision."
    )
    what = "Assess robustness of clinical management decisions to record formatting"


class PR26ControlScenario(PR26BaseScenario):
    """PacificMedRobustness PR26 Control: whitespace-only changes, used as the noise floor."""

    name = "pr26_control"
    display_name = "Control"
    description = "Noise floor: whitespace and line-break changes only, with no clinical content change."
    main_metric = "medrobustness_control_failure_rate"
    what = "Measure the noise floor of the paired protocol using whitespace-only changes"


class PR26PositionScenario(PR26BaseScenario):
    """PacificMedRobustness PR26 Position: decisive facts scattered across the record."""

    name = "pr26_position"
    display_name = "Position"
    description = "Evaluates whether the decisive facts stay answerable when they are scattered across the record."
    what = "Assess robustness of clinical management decisions to scattering of decisive facts"


class PR26ParaphraseScenario(PR26BaseScenario):
    """PacificMedRobustness PR26 Paraphrase: question phrasing and register."""

    name = "pr26_paraphrase"
    display_name = "Paraphrase"
    description = (
        "Evaluates whether a clinically equivalent question, phrased in a different register, produces the same "
        "correct management decision."
    )
    what = "Assess robustness of clinical management decisions to question phrasing"


class PR26NegationScenario(PR26BaseScenario):
    """PacificMedRobustness PR26 Negation: findings restated as absence, plus inverted controls."""

    name = "pr26_negation"
    display_name = "Negation"
    description = "Evaluates whether a decisive finding restated as absence, or inverted, is handled consistently."
    what = "Assess robustness of clinical management decisions to negated and restated findings"


class PR26TemporalScenario(PR26BaseScenario):
    """PacificMedRobustness PR26 Temporal: timing re-expressed relative to the note time."""

    name = "pr26_temporal"
    display_name = "Temporal"
    description = (
        "Evaluates whether clinical timing expressed relative to the note time produces the same correct management "
        "decision."
    )
    what = "Assess robustness of clinical management decisions to temporal re-expression"


class PR26NumeracyScenario(PR26BaseScenario):
    """PacificMedRobustness PR26 Numeracy: decisive quantities re-expressed or derived."""

    name = "pr26_numeracy"
    display_name = "Numeracy"
    description = (
        "Evaluates whether a decisive quantity re-expressed in another unit or derived from other values produces "
        "the same correct management decision."
    )
    what = "Assess robustness of clinical management decisions to numeric re-expression"


class PR26TerminologyScenario(PR26BaseScenario):
    """PacificMedRobustness PR26 Terminology: abbreviations and brand names for generic drugs."""

    name = "pr26_terminology"
    display_name = "Terminology"
    description = (
        "Evaluates whether a clinical abbreviation expanded in the record, or a generic drug name replaced by its "
        "brand name, produces the same correct management decision."
    )
    what = "Assess robustness of clinical management decisions to medical terminology variants"
