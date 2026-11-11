import json
import re
import time
from dataclasses import dataclass
from string import Template
from typing import Dict, Optional, TypedDict, Union, Callable, Any, Set

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.request import Request
from helm.common.response_format import ResponseFormat


class AnnotatorResponseParseFailure(Exception):
    def __init__(self, response_text: str, **kwargs):
        self.response_text = response_text
        super().__init__(kwargs)


@dataclass
class AnnotatorModelInfo:
    model_name: str
    model_deployment: str


def score_with_reasoning_with_gpt_and_llama(
    auto_client: AutoClient,
    annotator_prompt: str,
) -> Dict[str, Optional[Union[str, float]]]:
    """EXPERIMENTAL: DO NOT USE IN PRODUCTION

    Score using GPT-4o and Llama 3.1 for safety scenarios in HELM Safety."""
    # TODO: Make this configurable
    SHORT_NAME_TO_MODEL_INFO: Dict[str, AnnotatorModelInfo] = {
        "gpt": AnnotatorModelInfo(model_name="openai/gpt-4o-2024-05-13", model_deployment="openai/gpt-4o-2024-05-13"),
        "llama": AnnotatorModelInfo(
            model_name="meta/llama-3.1-405b-instruct-turbo", model_deployment="together/llama-3.1-405b-instruct-turbo"
        ),
    }
    result: Dict[str, Optional[Union[str, float]]] = {"prompt_text": annotator_prompt}
    for short_name, model_info in SHORT_NAME_TO_MODEL_INFO.items():
        try:
            score_and_reasoning = score_with_reasoning(
                auto_client,
                annotator_prompt,
                annotator_model=model_info.model_name,
                annotator_model_deployment=model_info.model_deployment,
            )
            result[f"{short_name}_score"] = score_and_reasoning["score"]
            result[f"{short_name}_reasoning"] = score_and_reasoning["reasoning"]
        except AnnotatorResponseParseFailure as e:
            result[f"{short_name}_score"] = None
            result[f"{short_name}_reasoning"] = None
            result[f"{short_name}_response_text"] = e.response_text
    return result


class ScoreAndReasoning(TypedDict):
    score: float
    reasoning: str


def score_with_reasoning(
    auto_client: AutoClient,
    annotator_prompt: str,
    annotator_model: str,
    annotator_model_deployment: str,
) -> ScoreAndReasoning:
    annotator_request = Request(
        model=annotator_model,
        model_deployment=annotator_model_deployment,
        prompt=annotator_prompt,
        temperature=0.0,
        max_tokens=256,
    )
    annotator_response = auto_client.make_request(annotator_request)
    if not annotator_response.success:
        raise Exception(f"Annotation request failed: {annotator_response.error}")
    assert len(annotator_response.completions) == 1
    annotator_response_text = annotator_response.completions[0].text

    # fuzzy match regex check, allows for different casing, or forgetting / in end tag
    reasoning_match = re.search(
        r"<\s*reasoning\s*>(.*?)<\/?\s*reasoning\s*>", annotator_response_text, re.DOTALL | re.IGNORECASE
    )
    score_match = re.search(r"<\s*score\s*>(.*?)<\/?\s*score\s*>", annotator_response_text, re.DOTALL | re.IGNORECASE)
    if not reasoning_match or not score_match:
        raise AnnotatorResponseParseFailure(
            message=f"Could not parse markup in raw response: '{annotator_response_text}'",
            response_text=annotator_response_text,
        )
    reasoning = reasoning_match.group(1).strip()
    try:
        score = float(score_match.group(1).strip())
    except ValueError:
        raise AnnotatorResponseParseFailure(
            message=f"Could not parse score as float from raw request: '{annotator_response_text}'",
            response_text=annotator_response_text,
        )

    return {"reasoning": reasoning, "score": score}


class LLMAsJuryAnnotator(Annotator):
    """
    A flexible LLM-based annotator that can be configured for different annotation scenarios.

    This annotator supports:
    - Custom prompt templates
    - Multiple evaluation models
    - Configurable evaluation criteria
    - Robust error handling
    """

    def __init__(
        self,
        name: str,
        auto_client: AutoClient,
        prompt_template: str,
        annotation_criteria: Dict[str, Set[str]],
        annotator_models: Dict[str, AnnotatorModelInfo],
        preprocessor: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the LLMAsJuryAnnotator.

        :param auto_client: Client for making API requests
        :param prompt_template: Template for generating prompts
        :param annotation_criteria: Dictionary defining expected annotation structure
        :param annotator_models: Dictionary of models to use for annotation
        :param preprocessor: Optional function to preprocess model responses
        """
        self.name = name
        self._auto_client = auto_client
        self._prompt_template = prompt_template
        self._annotation_criteria = annotation_criteria
        self._annotator_models = annotator_models
        self._preprocessor = preprocessor or self._sanitize_model_response

    def _sanitize_model_response(self, model_response: str) -> str:
        """
        Sanitize the model response to extract JSON.

        :param model_response: Raw model response
        :return: Extracted JSON string
        """
        json_match = re.search(r"\{.*\}", model_response, re.DOTALL)
        return json_match.group(0) if json_match else model_response

    def _interpolate_prompt(
        self, request_state: RequestState, custom_replacements: Optional[Dict[str, str]] = None
    ) -> str:
        """Interpolate prompt templates safely, supporting {{QUESTION}}-style files."""
        # Build required/optional fields
        replacements: Dict[str, str] = {
            "QUESTION": request_state.instance.input.text,
            "RESPONSE": (
                request_state.result.completions[0].text
                if request_state.result and request_state.result.completions
                else ""
            ),
            # GOLD is optional; keep empty if not present
            "GOLD_RESPONSE": (
                request_state.instance.references[0].output.text
                if getattr(request_state.instance, "references", None)
                else ""
            ),
        }
        if custom_replacements:
            replacements.update(custom_replacements)

        tmpl_text = self._prompt_template

        tmpl_text = (
            tmpl_text.replace("{QUESTION}", "$QUESTION")
            .replace("{RESPONSE}", "$RESPONSE")
            .replace("{GOLD_RESPONSE}", "$GOLD_RESPONSE")
        )

        return Template(tmpl_text).substitute(replacements)

    def _validate_annotation(
        self,
        annotator_criteria: Dict[str, Any],
        annotator_name: str,
    ) -> bool:
        """
        Recursively validate the annotation meets expected criteria.
        """

        def score_is_float(score_value: Any, score_path: str) -> bool:
            try:
                float(score_value)
            except (TypeError, ValueError):
                hwarn(
                    f"Invalid score at '{score_path}'. "
                    f"Expected a float-compatible value, got {score_value!r} "
                    f"from {annotator_name}."
                )
                return False
            return True

        def validate(schema: Dict[str, Any], data: Dict[str, Any], path: str = "") -> bool:
            for key, expected in schema.items():
                current_path = f"{path}.{key}" if path else key

                if key not in data:
                    hwarn(f"Missing key '{current_path}' in response from {annotator_name}.")
                    return False

                value = data[key]

                if key == "score" and not isinstance(expected, (dict, list, set)):
                    if not score_is_float(value, current_path):
                        return False

                # Case 1: nested dict → recurse
                if isinstance(expected, dict):
                    if not isinstance(value, dict):
                        hwarn(f"Expected dict at '{current_path}' but got {type(value)} " f"from {annotator_name}.")
                        return False
                    if not validate(expected, value, current_path):
                        return False

                # Case 2: list of required subkeys
                elif isinstance(expected, list):
                    for subkey in expected:
                        if subkey not in value:
                            hwarn(f"Missing subkey '{current_path}.{subkey}' " f"in response from {annotator_name}.")
                            return False
                        if subkey == "score" and not score_is_float(value[subkey], f"{current_path}.{subkey}"):
                            return False

                # Case 2b: set of required subkeys
                elif isinstance(expected, set):
                    if not isinstance(value, dict):
                        hwarn(f"Expected dict at '{current_path}' but got {type(value)} " f"from {annotator_name}.")
                        return False
                    for subkey in expected:
                        if subkey not in value:
                            hwarn(f"Missing subkey '{current_path}.{subkey}' " f"in response from {annotator_name}.")
                            return False
                        if subkey == "score" and not score_is_float(value[subkey], f"{current_path}.{subkey}"):
                            return False

                # Case 3: type checking (optional but useful)
                elif isinstance(expected, type):
                    if not isinstance(value, expected):
                        hwarn(
                            f"Invalid type at '{current_path}'. "
                            f"Expected {expected}, got {type(value)} "
                            f"from {annotator_name}."
                        )
                        return False

            return True

        return validate(self._annotation_criteria, annotator_criteria)

    def make_json_serializable(self, obj):
        if isinstance(obj, set):
            return [self.make_json_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.make_json_serializable(item) for item in obj]
        return obj

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        """
        Annotate the request state using multiple LLM models.

        :param request_state: The request state to annotate
        :return: Dictionary of annotations from different models
        """
        assert request_state.result
        assert len(request_state.result.completions) == 1

        # Check for empty model output
        model_output_text = request_state.result.completions[0].text
        if not model_output_text.strip():
            hwarn("Annotator skipped sending requests because the model response was empty")
            return {
                "prompt_text": None,
                "empty_output_equivalence_judgement": False,
            }

        # Prepare prompt
        annotator_prompt = self._interpolate_prompt(request_state)
        annotations: Dict[str, Union[Optional[str], Optional[bool], Dict[str, Any]]] = {"prompt_text": annotator_prompt}

        # Track failed annotations for each model
        failed_counts: Dict[str, int] = {name: 0 for name in self._annotator_models}

        # Annotate using multiple models
        for annotator_name, annotator_model_info in self._annotator_models.items():
            max_attempts = 3
            retry_delays = [1, 2, 4]

            for attempt in range(1, max_attempts + 1):
                current_prompt = annotator_prompt
                if attempt > 1:
                    current_prompt += """

        IMPORTANT: Your previous response was not valid JSON.

        For this attempt:
        - Return ONLY valid JSON.
        - USE **double quotes** (") for all keys and string values. DO NOT use single quotes (').
        - Ensure all keys and values are properly quoted and separated by commas.
        - Ensure all opening braces {{ have corresponding closing braces }}.
        - Ensure all opening brackets [ have corresponding closing brackets ].
        - Ensure all strings are properly escaped.
        - Do NOT use markdown code fences.
        - Do NOT include explanations or additional text.
        - The entire response must be a single valid JSON object.
        - Make sure all strings, quotes, commas, and braces are valid JSON syntax that folows the JSON Structure of the Output Format explained above.

        """
                    delay = retry_delays[attempt - 2]
                    hlog(
                        f"Retrying annotation for {annotator_name} in {delay}s "
                        f"(attempt {attempt}/{max_attempts})"
                        f"prompt: {current_prompt}"
                    )
                    time.sleep(delay)

                try:
                    annotator_criteria = self._annotate_with_model(
                        current_prompt,
                        annotator_model_info,
                        annotator_name,
                    )

                    if annotator_criteria is not None:
                        try:
                            annotations[annotator_name] = annotator_criteria
                            break
                        except (json.JSONDecodeError, TypeError):
                            hlog(
                                f"WARNING invalid JSON from {annotator_name} "
                                f"(attempt {attempt}/{max_attempts})"
                                f"response: {annotator_criteria}"
                            )
                    else:
                        hlog(
                            f"WARNING annotation returned None for {annotator_name} "
                            f"(attempt {attempt}/{max_attempts})"
                            f"prompt: {current_prompt}"
                        )

                except Exception as e:
                    hlog(
                        f"ERROR annotating with {annotator_name} "
                        f"(attempt {attempt}/{max_attempts}): {e}"
                        f"prompt: {current_prompt}"
                    )

                if attempt == max_attempts:
                    failed_counts[annotator_name] += 1

        hlog(f"Failed model annotations: {failed_counts}")
        return annotations

    def build_json_schema(self, annotation_criteria: Union[Dict[str, Any], Set[str]]) -> dict:
        """Build a JSON Schema from annotation criteria.

        Supported formats:
        - Set[str]: each criterion gets numeric score and string explanation fields.
        - Dict[str, Set[str] | list[str]]: each criterion gets the listed fields.
        - Dict[str, type]: each key is a top-level field with the declared type.
        - Dict[str, dict]: each criterion is a nested object; value types are
        inferred from the example values.
        """

        print(f"Building JSON Schema from annotation criteria: {annotation_criteria}")
        type_map = {
            str: "string",
            bool: "boolean",
            int: "integer",
            float: "number",
        }

        if isinstance(annotation_criteria, set):
            criteria = sorted(annotation_criteria)
            properties = {
                criterion: {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["score", "explanation"],
                    "additionalProperties": False,
                }
                for criterion in criteria
            }

        elif isinstance(annotation_criteria, dict):
            properties = {}

            for name, spec in annotation_criteria.items():
                # Top-level field with an explicit Python type.
                if spec is str or spec is bool or spec is int or spec is float:
                    properties[name] = {"type": type_map[spec]}
                    continue

                # Nested criterion with a set/list of field names.
                if isinstance(spec, (set, list)):
                    fields = sorted(spec) if isinstance(spec, set) else list(spec)
                    criterion_properties = {
                        field: {"type": "number" if field == "score" else "string"} for field in fields
                    }

                # Nested criterion with field example values.
                elif isinstance(spec, dict):
                    fields = list(spec)
                    criterion_properties = {}

                    for field, example in spec.items():
                        if field == "score":
                            field_type = "number"
                        elif isinstance(example, bool):
                            field_type = "boolean"
                        elif isinstance(example, int):
                            field_type = "integer"
                        elif isinstance(example, float):
                            field_type = "number"
                        elif isinstance(example, str):
                            field_type = "string"
                        elif isinstance(example, list):
                            field_type = "array"
                        elif isinstance(example, dict):
                            field_type = "object"
                        elif example is None:
                            raise ValueError(f"Cannot infer a schema type from None for {name}.{field}")
                        else:
                            raise ValueError(
                                f"Unsupported example type for {name}.{field}: " f"{type(example).__name__}"
                            )

                        criterion_properties[field] = {"type": field_type}

                else:
                    raise ValueError(f"Unsupported annotation specification for {name!r}: " f"{spec!r}")

                properties[name] = {
                    "type": "object",
                    "properties": criterion_properties,
                    "required": fields,
                    "additionalProperties": False,
                }

        else:
            raise ValueError("annotation_criteria must be a dict or set, " f"got {type(annotation_criteria).__name__}")

        json_schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        print(f"Built JSON Schema: {json.dumps(json_schema, indent=2)}")
        return json_schema

    def _annotate_with_model(
        self, prompt: str, model_info: AnnotatorModelInfo, annotator_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Annotate using a specific model with enhanced JSON parsing.

        :param prompt: Interpolated prompt
        :param model_info: Model information
        :param annotator_name: Name of the annotator
        :return: Annotation criteria or None if failed
        """
        print(f"Annotating with {model_info}")
        annotator_request = Request(
            model=model_info.model_name,
            model_deployment=model_info.model_deployment,
            prompt=prompt,
            temperature=0.0,
            max_tokens=4096,
            response_format=ResponseFormat(json_schema=self.build_json_schema(self._annotation_criteria)),
            # Ensure the model responds with JSON
        )

        annotator_response = self._auto_client.make_request(annotator_request)

        if not annotator_response.success:
            hwarn(f"Got an error response from {model_info.model_name}: " f"{annotator_response.error}")
            return None

        try:
            annotator_output = annotator_response.completions[0].text
            print(f"Raw model output from {model_info.model_name}: {annotator_output}")
            annotator_output = self._preprocessor(annotator_output)
            print(f"Preprocessed model output from {model_info.model_name}: {annotator_output}")

            try:
                annotator_criteria = json.loads(annotator_output)
            except json.JSONDecodeError as e:
                if e.msg == "Expecting ',' delimiter":
                    # Attempt to fix incomplete JSON by adding a closing brace
                    annotator_output = annotator_output + "}"
                    try:
                        annotator_criteria = json.loads(annotator_output)
                    except Exception as ex:
                        hwarn(
                            f"Error parsing response from {model_info.model_name} "
                            f"after adding closing brace: {ex}. "
                            f"Model output: {annotator_output}"
                        )
                        return None
                else:
                    # For other JSON decoding errors
                    hwarn(
                        f"JSON decoding error from {model_info.model_name}: {e}. " f"Model output: {annotator_output}"
                    )
                    return None

            # Validate annotation structure
            if not self._validate_annotation(annotator_criteria, annotator_name):
                hwarn(
                    f"Annotation from {model_info.model_name} did not meet expected criteria. "
                    f"Model output: {annotator_output}"
                )
                return None

            return annotator_criteria

        except Exception as e:
            hwarn(
                f"Unexpected error processing response from {model_info.model_name}: {e}. "
                f"Model output: {annotator_output}"
            )
            return None
