import json
from typing import Any, Dict, List, Optional

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Instance,
    Output,
    PassageQuestionInput,
    Reference,
    Scenario,
    ScenarioMetadata,
)
from helm.common.general import check_file_exists


class CPCQAScenario(Scenario):
    """
    From "CPC-Bench: Benchmarking Large Language Models on Clinicopathological Conferences"
    (https://arxiv.org/pdf/2509.12194), CPC-Bench is built from the New England Journal of
    Medicine Clinicopathological Conference (CPC) case records. This scenario implements the
    Question Answering (QA) task, which probes clinical concept understanding and factual
    knowledge grounded in a patient case presentation.

    The full benchmark covers 7,102 cases; a public subset of 100 cases is distributed at
    https://cpcbench.com/dataset_100.html. The cases are licensed from NEJM and are not
    redistributed with MedHELM, so the dataset must be downloaded separately and passed to
    this scenario via `data_path`.

    The data file is a JSON list of question records. The following is an example
    from the dataset:

    ```
    {
      "id": "NEJM200003303421308",
      "title": "Case 10-2000",
      "publication_date": "2000-03-30",
      "publication_year": 2000,
      "presentation_of_case": "A 63-year-old man was admitted to the hospital in early July
        because of rapidly progressive changes in behavior and ataxia. ...",
      "question": "What term best describes the 13-day development of behavioral changes,
        headache, fever, and confusion seen in this patient?",
      "options": {
        "A": "Chronic encephalopathy",
        "B": "Subacute meningoencephalitis",
        "C": "Acute encephalopathy",
        "D": "Delirium tremens",
        "E": "Acute confusional state secondary to metabolic derangement"
      },
      "correct_answer": "C",
      "reasoning_snippet": "The syndrome he had at that time is usually referred to as acute
        encephalopathy, which encompasses many diffuse and multifocal disorders that affect
        both cerebral hemispheres.",
      "question_num": 1
    }
    ```

    Citation

    ```
    @misc{cpcbench2025,
      title={Clinicopathological Conference Benchmark (CPC-Bench)},
      url={https://arxiv.org/abs/2509.12194},
      publisher={arXiv},
      year={2025}
    }
    ```
    """

    name = "cpc_qa"
    description = (
        "CPC-QA is a clinical question-answering benchmark built from New England Journal of "
        "Medicine Clinicopathological Conference case records. Each multiple-choice question is "
        "grounded in a full patient case presentation and assesses a model's ability to interpret "
        "the case and apply clinical concept and factual knowledge."
    )
    tags = ["question_answering", "biomedical", "medicine"]

    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path

    def get_instances(self, output_path: str) -> List[Instance]:
        check_file_exists(self.data_path, msg=f"[CPCQAScenario] Required data file not found: '{self.data_path}'")

        with open(self.data_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"[CPCQAScenario] Expected a JSON list of records in '{self.data_path}'")

        instances: List[Instance] = []
        for record in records:
            instance = self._create_instance(record)
            if instance is not None:
                instances.append(instance)

        return instances

    def _create_instance(self, record: Dict[str, Any]) -> Optional[Instance]:
        presentation: str = record.get("presentation_of_case", "")
        question: str = record.get("question", "")
        options: Dict[str, str] = record.get("options", {})
        correct_answer: str = record.get("correct_answer", "")

        if not presentation or not question or not options or correct_answer not in options:
            return None

        references: List[Reference] = [
            Reference(Output(text=options[letter]), tags=[CORRECT_TAG] if letter == correct_answer else [])
            for letter in sorted(options)
        ]
        prompt = PassageQuestionInput(
            passage=presentation,
            # The trailing newline separates the question from the lettered answer
            # choices that the multiple-choice adapter appends.
            question=question + "\n",
            passage_prefix="Case presentation: ",
            separator="\n\n",
        )
        return Instance(input=prompt, references=references, split=TEST_SPLIT)

    def get_metadata(self):
        return ScenarioMetadata(
            name="cpc_qa",
            display_name="CPC-QA",
            description="CPC-QA is a clinical question-answering benchmark built from New England "
            "Journal of Medicine Clinicopathological Conference case records. Each multiple-choice "
            "question is grounded in a full patient case presentation and assesses a model's ability "
            "to interpret the case and apply clinical concept and factual knowledge.",
            taxonomy=TaxonomyInfo(
                task="Question answering",
                what="Answer questions based on patient case presentation",
                when="Any",
                who="Researcher",
                language="English",
            ),
            main_metric="exact_match",
            main_split="test",
        )
