import json
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import pytest

from helm.benchmark.scenarios.cpc_qa_scenario import CPCQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Instance


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

PRESENTATION = "A 63-year-old man was admitted because of progressive changes in behavior and ataxia."
QUESTION = "What term best describes the 13-day development of behavioral changes and confusion?"
OPTIONS = {
    "A": "Chronic encephalopathy",
    "B": "Subacute meningoencephalitis",
    "C": "Acute encephalopathy",
    "D": "Delirium tremens",
    "E": "Acute confusional state secondary to metabolic derangement",
}


def _record(**overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "id": "NEJM200003303421308",
        "title": "Case 10-2000",
        "publication_year": 2000,
        "presentation_of_case": PRESENTATION,
        "question": QUESTION,
        "options": dict(OPTIONS),
        "correct_answer": "C",
        "question_num": 1,
    }
    record.update(overrides)
    return record


def _instances(payload: Any) -> List[Instance]:
    with TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "dataset_100.json")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return CPCQAScenario(data_path=data_path).get_instances(tmp)


# ---------------------------------------------------------------------------
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = CPCQAScenario(data_path="/tmp/dataset_100.json")
    assert scenario.data_path == "/tmp/dataset_100.json"


def test_class_attributes():
    assert CPCQAScenario.name == "cpc_qa"
    assert CPCQAScenario.tags == ["question_answering", "biomedical", "medicine"]
    assert "CPC-QA" in CPCQAScenario.description


# ---------------------------------------------------------------------------
# Instance construction.
# ---------------------------------------------------------------------------


def test_single_record_produces_one_test_split_instance():
    instances = _instances([_record()])

    assert len(instances) == 1
    assert instances[0].split == TEST_SPLIT


def test_input_contains_case_presentation_and_question():
    text = _instances([_record()])[0].input.text

    assert text.startswith("Case presentation: ")
    assert PRESENTATION in text
    assert QUESTION in text
    # PassageQuestionInput's default question prefix, with the configured separator.
    assert "\n\nQuestion: " in text
    # The question must end with a newline so the adapter's "A. " choice does not
    # get glued onto the end of the question text.
    assert text.endswith(QUESTION + "\n")


def test_references_are_option_text_in_sorted_letter_order():
    references = _instances([_record()])[0].references

    assert [reference.output.text for reference in references] == [
        OPTIONS[letter] for letter in ["A", "B", "C", "D", "E"]
    ]


def test_exactly_one_reference_is_tagged_correct_and_holds_option_text():
    """The correct reference must carry the option *text*, not the letter — the
    multiple-choice adapter renders the A/B/C/D/E labels itself."""
    references = _instances([_record()])[0].references
    correct = [reference for reference in references if CORRECT_TAG in reference.tags]

    assert len(correct) == 1
    assert correct[0].output.text == OPTIONS["C"]
    assert correct[0].output.text != "C"


def test_multiple_records_from_the_same_case_each_become_an_instance():
    payload = [
        _record(question_num=1),
        _record(question_num=2, question="Which test is most likely to be diagnostic?", correct_answer="A"),
    ]

    assert len(_instances(payload)) == 2


def test_empty_list_produces_no_instances():
    assert _instances([]) == []


# ---------------------------------------------------------------------------
# Malformed records are skipped, not fatal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"correct_answer": "Z"},  # answer letter not among the options
        {"correct_answer": ""},  # missing answer
        {"options": {}},  # no options
        {"question": ""},  # missing question
        {"presentation_of_case": ""},  # missing case text
    ],
)
def test_malformed_record_is_skipped(overrides: Dict[str, Any]):
    assert _instances([_record(**overrides)]) == []


def test_malformed_record_does_not_drop_the_valid_ones():
    payload = [_record(), _record(correct_answer="Z"), _record(question_num=3)]

    assert len(_instances(payload)) == 2


# ---------------------------------------------------------------------------
# File-level errors.
# ---------------------------------------------------------------------------


def test_non_list_json_raises_value_error():
    with pytest.raises(ValueError, match="Expected a JSON list"):
        _instances({"id": "NEJM1"})


def test_missing_file_raises_file_not_found():
    with TemporaryDirectory() as tmp:
        scenario = CPCQAScenario(data_path=os.path.join(tmp, "absent.json"))
        with pytest.raises(FileNotFoundError, match="CPCQAScenario"):
            scenario.get_instances(tmp)


# ---------------------------------------------------------------------------
# Metadata — must stay in sync with schema_medhelm.yaml.
# ---------------------------------------------------------------------------


def test_metadata_matches_schema_entry():
    metadata = CPCQAScenario(data_path="/tmp/dataset_100.json").get_metadata()

    assert metadata.name == "cpc_qa"
    assert metadata.display_name == "CPC-QA"
    assert metadata.main_metric == "exact_match"
    assert metadata.main_split == "test"
    assert metadata.taxonomy.task == "Question answering"
    assert metadata.taxonomy.what == "Answer questions based on patient case presentation"
    assert metadata.taxonomy.who == "Researcher"
    assert metadata.taxonomy.when == "Any"
    assert metadata.taxonomy.language == "English"
