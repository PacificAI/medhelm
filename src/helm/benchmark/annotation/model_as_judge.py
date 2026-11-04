import json
import math
import re
import time
from dataclasses import dataclass
from string import Template
from typing import Any, Callable, Dict, Optional, Set, TypedDict, Union

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.annotator import Annotator
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hlog, hwarn
from helm.common.request import Request
from helm.common.response_format import ResponseFormat


class AnnotatorResponseParseFailure(Exception):
    """Raised when a model response cannot be parsed into its expected format."""

    def __init__(self, response_text: str, message: str = "Could not parse model response") -> None:
        self.response_text = response_text
        super().__init__(message)


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
        annotation_criteria: Union[Dict[str, Any], Set[str]],
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
        """Extract the first parseable JSON object from a possibly wrapped response."""
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", model_response):
            try:
                parsed, end = decoder.raw_decode(model_response[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return model_response[match.start() : match.start() + end]
        return model_response

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

    def _validate_annotation(self, annotator_criteria: Any, annotator_name: str) -> bool:
        """Validate the model-produced annotation against the configured criteria.

        Malformed model output is treated as a validation failure, not as an
        exception. This is important because the caller uses a ``False`` result
        to trigger a retry.
        """

        def score_is_numeric(score_value: Any, score_path: str) -> bool:
            # bool is a subclass of int in Python, but it is not a meaningful score.
            if isinstance(score_value, bool):
                hwarn(
                    f"Invalid score at '{score_path}'. Expected a finite numeric value, "
                    f"got {score_value!r} from {annotator_name}."
                )
                return False
            try:
                numeric_score = float(score_value)
            except (TypeError, ValueError, OverflowError):
                hwarn(
                    f"Invalid score at '{score_path}'. Expected a finite numeric value, "
                    f"got {score_value!r} from {annotator_name}."
                )
                return False
            if not math.isfinite(numeric_score):
                hwarn(
                    f"Invalid score at '{score_path}'. Scores must be finite, "
                    f"got {score_value!r} from {annotator_name}."
                )
                return False
            return True

        def validate(schema: Dict[str, Any], data: Any, path: str = "") -> bool:
            if not isinstance(data, dict):
                hwarn(
                    f"Expected an object at '{path or '<root>'}' but got "
                    f"{type(data).__name__} from {annotator_name}."
                )
                return False

            for key, expected in schema.items():
                current_path = f"{path}.{key}" if path else key
                if key not in data:
                    hwarn(f"Missing key '{current_path}' in response from {annotator_name}.")
                    return False

                value = data[key]

                # Score values are intentionally float-compatible, regardless of
                # whether the example/schema marker was int or float.
                if key == "score":
                    if not score_is_numeric(value, current_path):
                        return False
                    continue

                # A nested schema requires a JSON object.
                if isinstance(expected, dict):
                    if not isinstance(value, dict):
                        hwarn(
                            f"Expected an object at '{current_path}' but got "
                            f"{type(value).__name__} from {annotator_name}."
                        )
                        return False
                    if not validate(expected, value, current_path):
                        return False

                # A list or set describes required keys within an object.
                elif isinstance(expected, (list, set)):
                    if not isinstance(value, dict):
                        hwarn(
                            f"Expected an object at '{current_path}' for required subkeys, "
                            f"but got {type(value).__name__} from {annotator_name}."
                        )
                        return False
                    for subkey in expected:
                        subkey_path = f"{current_path}.{subkey}"
                        if subkey not in value:
                            hwarn(f"Missing subkey '{subkey_path}' in response from {annotator_name}.")
                            return False
                        if subkey == "score" and not score_is_numeric(value[subkey], subkey_path):
                            return False

                # A type object in the criteria is used as an explicit type check.
                elif isinstance(expected, type):
                    # Avoid accepting bool where an integer is expected.
                    type_matches = isinstance(value, expected)
                    if expected is int and isinstance(value, bool):
                        type_matches = False
                    if not type_matches:
                        hwarn(
                            f"Invalid type at '{current_path}'. Expected {expected.__name__}, "
                            f"got {type(value).__name__} from {annotator_name}."
                        )
                        return False

            return True

        if isinstance(self._annotation_criteria, set):
            # A top-level set means each named criterion has score/explanation.
            schema: Dict[str, Any] = {
                criterion: {"score": float, "explanation": str} for criterion in self._annotation_criteria
            }
        elif isinstance(self._annotation_criteria, dict):
            schema = self._annotation_criteria
        else:
            hwarn(
                "Invalid annotation criteria configuration: expected a dict or set, "
                f"got {type(self._annotation_criteria).__name__}."
            )
            return False

        return validate(schema, annotator_criteria)

    def make_json_serializable(self, obj):
        if isinstance(obj, set):
            return [self.make_json_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.make_json_serializable(item) for item in obj]
        return obj

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        """Annotate one model response, retrying malformed or invalid judge output."""
        if not request_state.result or not request_state.result.completions:
            hwarn("Annotator skipped sending requests because the original model response was missing")
            return {
                "prompt_text": None,
                "empty_output_equivalence_judgement": False,
            }
        if len(request_state.result.completions) != 1:
            hwarn(
                "Annotator expected exactly one completion from the evaluated model, "
                f"got {len(request_state.result.completions)}"
            )
            return {
                "prompt_text": None,
                "empty_output_equivalence_judgement": False,
            }

        model_output_text = request_state.result.completions[0].text
        if not model_output_text.strip():
            hwarn("Annotator skipped sending requests because the model response was empty")
            return {
                "prompt_text": None,
                "empty_output_equivalence_judgement": False,
            }

        annotator_prompt = self._interpolate_prompt(request_state)
        annotations: Dict[str, Any] = {"prompt_text": annotator_prompt}
        failed_counts: Dict[str, int] = {name: 0 for name in self._annotator_models}

        max_attempts = 3
        retry_delays = (1, 2)  # One delay before each retry; no unused delay entries.
        retry_instruction = """

IMPORTANT: Your previous response could not be parsed or did not match the required schema.
For this attempt:
- Return only one valid JSON object, with no Markdown fences or extra commentary.
- Use double quotes for all JSON keys and string values.
- Include every required key and use the expected value types.
- Ensure all strings, quotes, commas, braces, and brackets are valid JSON.
"""

        for annotator_name, annotator_model_info in self._annotator_models.items():
            succeeded = False
            for attempt in range(1, max_attempts + 1):
                if attempt == 1:
                    current_prompt = annotator_prompt
                else:
                    delay = retry_delays[min(attempt - 2, len(retry_delays) - 1)]
                    hlog(
                        f"Retrying annotation for {annotator_name} after {delay}s "
                        f"(attempt {attempt}/{max_attempts})"
                    )
                    time.sleep(delay)
                    current_prompt = annotator_prompt + retry_instruction

                try:
                    annotator_criteria = self._annotate_with_model(
                        current_prompt,
                        annotator_model_info,
                        annotator_name,
                    )
                    if annotator_criteria is None:
                        hwarn(
                            f"Annotation from {annotator_name} was invalid or unavailable "
                            f"(attempt {attempt}/{max_attempts})."
                        )
                        continue

                    # Defensive validation at the retry boundary means alternate or
                    # overridden implementations cannot bypass the output contract.
                    if not isinstance(annotator_criteria, dict):
                        hwarn(
                            f"Expected an object from {annotator_name}, got "
                            f"{type(annotator_criteria).__name__} "
                            f"(attempt {attempt}/{max_attempts})."
                        )
                        continue
                    if not self._validate_annotation(annotator_criteria, annotator_name):
                        hwarn(
                            f"Annotation from {annotator_name} failed schema validation "
                            f"(attempt {attempt}/{max_attempts})."
                        )
                        continue

                    annotations[annotator_name] = annotator_criteria
                    succeeded = True
                    break
                except Exception as exc:
                    # Keep the task alive if one judge fails. Avoid logging the full
                    # prompt/clinical content, which may contain sensitive data.
                    hwarn(
                        f"Error annotating with {annotator_name} "
                        f"(attempt {attempt}/{max_attempts}): {type(exc).__name__}: {exc}"
                    )

            if not succeeded:
                failed_counts[annotator_name] += 1

        hlog(f"Failed model annotations: {failed_counts}")
        return annotations

    def build_json_schema(self, annotation_criteria: Union[dict, set]) -> dict:
        """Build a JSON Schema compatible with the configured annotation criteria.

        Scores use JSON Schema ``number`` rather than ``integer`` so fractional
        scores remain valid and align with float-compatible score validation.
        """
        if isinstance(annotation_criteria, set):
            criteria = sorted(annotation_criteria)
            return {
                "type": "object",
                "properties": {
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
                },
                "required": criteria,
                "additionalProperties": False,
            }

        if not isinstance(annotation_criteria, dict):
            raise ValueError("annotation_criteria must be a dict or set, " f"got {type(annotation_criteria).__name__}")

        properties: Dict[str, Any] = {}
        criterion_properties: Dict[str, Any] = {}
        for criterion, value in annotation_criteria.items():
            if isinstance(value, (set, list)):
                fields = sorted(value) if isinstance(value, set) else value
                criterion_properties = {field: {"type": "number" if field == "score" else "string"} for field in fields}
            elif isinstance(value, dict):
                for field, example_value in value.items():
                    if field == "score":
                        field_type = "number"
                    elif isinstance(example_value, bool):
                        field_type = "boolean"
                    elif isinstance(example_value, int):
                        field_type = "integer"
                    elif isinstance(example_value, float):
                        field_type = "number"
                    elif isinstance(example_value, str):
                        field_type = "string"
                    elif isinstance(example_value, list):
                        field_type = "array"
                    elif isinstance(example_value, dict):
                        field_type = "object"
                    elif example_value is None:
                        field_type = "string"
                    else:
                        raise ValueError(
                            f"Unsupported type for '{criterion}.{field}': " f"{type(example_value).__name__}"
                        )
                    criterion_properties[field] = {"type": field_type}
            else:
                raise ValueError(
                    f"Expected '{criterion}' to contain a dictionary, list, or set, " f"got {type(value).__name__}"
                )

            properties[criterion] = {
                "type": "object",
                "properties": criterion_properties,
                "required": list(criterion_properties.keys()),
                "additionalProperties": False,
            }

        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        }

    def _annotate_with_model(
        self, prompt: str, model_info: AnnotatorModelInfo, annotator_name: str
    ) -> Optional[Dict[str, Any]]:
        """Call a judge model, parse its JSON, and validate the result.

        Any invalid JSON or invalid annotation structure returns ``None``. The
        caller interprets that as a failed attempt and retries.
        """
        annotator_request = Request(
            model=model_info.model_name,
            model_deployment=model_info.model_deployment,
            prompt=prompt,
            temperature=0.0,
            max_tokens=4096,
            response_format=ResponseFormat(json_schema=self.build_json_schema(self._annotation_criteria)),
        )

        annotator_response = self._auto_client.make_request(annotator_request)
        if not annotator_response.success:
            hwarn(f"Annotation request failed for {model_info.model_name}: " f"{annotator_response.error}")
            return None

        if not annotator_response.completions:
            hwarn(f"Empty completion list from annotation model {model_info.model_name}.")
            return None

        annotator_output = annotator_response.completions[0].text
        try:
            processed_output = self._preprocessor(annotator_output)
            annotator_criteria = json.loads(processed_output)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            hwarn(f"Invalid JSON from {model_info.model_name}: {exc}. " f"Raw response: {annotator_output}")
            return None
        except Exception as exc:
            hwarn(f"Could not preprocess response from {model_info.model_name}: " f"{type(exc).__name__}: {exc}")
            return None

        if not isinstance(annotator_criteria, dict):
            hwarn(
                f"Expected a JSON object from {model_info.model_name}, got "
                f"{type(annotator_criteria).__name__}. Raw response: {annotator_output}"
            )
            return None

        if not self._validate_annotation(annotator_criteria, annotator_name):
            hwarn(
                f"Annotation from {model_info.model_name} failed schema validation. "
                f"Raw response: {annotator_output}"
            )
            return None

        return annotator_criteria
