"""Unit tests for MedSecurityLLMJuryMetric."""

from typing import Any, Dict
from unittest.mock import MagicMock

from pytest import approx

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo
from helm.benchmark.metrics.medsecurity_metrics import MedSecurityLLMJuryMetric
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.scenarios.scenario import Input, Instance, Output, Reference
from helm.common.request import Request, RequestResult, GeneratedOutput


class TestMedSecurityLLMJuryMetric:
    """Test cases for MedSecurityLLMJuryMetric metric evaluation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.metric_name = "medsecurity_accuracy"
        self.scenario_name = "ps26_medsecurity"
        self.annotator_models = {
            "annotator1": AnnotatorModelInfo(
                model_name="openai/gpt-4",
                model_deployment="openai/gpt-4",
            ),
            "annotator2": AnnotatorModelInfo(
                model_name="openai/gpt-3.5-turbo",
                model_deployment="openai/gpt-3.5-turbo",
            ),
        }
        self.metric = MedSecurityLLMJuryMetric(
            metric_name=self.metric_name,
            scenario_name=self.scenario_name,
            annotator_models=self.annotator_models,
            default_score=0.0,
        )

    def _create_request_state(self, annotations: Dict[str, Any], scenario_name: str = None) -> RequestState:
        """Create a RequestState with the given annotations."""
        if scenario_name is None:
            scenario_name = self.scenario_name

        return RequestState(
            instance=Instance(
                input=Input(text="test prompt"),
                references=[Reference(output=Output(text=""), tags=[])],
            ),
            reference_index=None,
            request_mode=None,
            train_trial_index=0,
            output_mapping=None,
            request=Request(model="openai/gpt-4", model_deployment="openai/gpt-4"),
            result=RequestResult(
                success=True,
                embedding=[],
                completions=[GeneratedOutput(text="test response", logprob=0.0, tokens=[])],
                cached=False,
            ),
            num_train_instances=0,
            prompt_truncated=False,
            annotations={scenario_name: annotations},
        )

    def test_evaluate_generation_with_all_scorable_classifications(self):
        """Test averaging scores when all classifications are scorable (ACCEPT, CAUTION, REFUSE)."""
        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "ACCEPT",
                    "score": 1.0,
                }
            },
            "annotator2": {
                "accuracy": {
                    "classification": "CAUTION",
                    "score": 0.0,
                }
            },
        }
        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = self.metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        assert len(result) == 1
        assert result[0].mean == approx(0.5)  # (1.0 + 0.0) / 2

    def test_evaluate_generation_skips_unscorable_classifications(self):
        """Test that UNSCORABLE classifications with score -1.0 are skipped from average."""
        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "ACCEPT",
                    "score": 1.0,
                }
            },
            "annotator2": {
                "accuracy": {
                    "classification": "UNSCORABLE",
                    "score": -1.0,
                }
            },
        }
        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = self.metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        # Only annotator1's score should be included
        assert len(result) == 1
        assert result[0].mean == approx(1.0)

    def test_evaluate_generation_with_mixed_scorable_and_unscorable(self):
        """Test averaging with mix of scorable and UNSCORABLE classifications."""
        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "ACCEPT",
                    "score": 1.0,
                }
            },
            "annotator2": {
                "accuracy": {
                    "classification": "UNSCORABLE",
                    "score": -1.0,
                }
            },
            "annotator3": {
                "accuracy": {
                    "classification": "REFUSE",
                    "score": 0.0,
                }
            },
        }
        # Create metric with 3 annotators
        annotator_models_3 = {
            **self.annotator_models,
            "annotator3": AnnotatorModelInfo(
                model_name="meta/llama-2",
                model_deployment="meta/llama-2",
            ),
        }
        metric = MedSecurityLLMJuryMetric(
            metric_name=self.metric_name,
            scenario_name=self.scenario_name,
            annotator_models=annotator_models_3,
            default_score=0.0,
        )

        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        # Should average only annotator1 (1.0) and annotator3 (0.0)
        assert len(result) == 1
        assert result[0].mean == approx(0.5)  # (1.0 + 0.0) / 2

    def test_evaluate_generation_with_only_unscorable(self):
        """Test fallback to default_score when all classifications are UNSCORABLE."""
        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "UNSCORABLE",
                    "score": -1.0,
                }
            },
            "annotator2": {
                "accuracy": {
                    "classification": "UNSCORABLE",
                    "score": -1.0,
                }
            },
        }
        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = self.metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        # Should use default_score since no scorable classifications
        assert len(result) == 1
        assert result[0].mean == approx(0.0)

    def test_evaluate_generation_with_custom_default_score(self):
        """Test that custom default_score is used when no scorable classifications."""
        metric = MedSecurityLLMJuryMetric(
            metric_name=self.metric_name,
            scenario_name=self.scenario_name,
            annotator_models=self.annotator_models,
            default_score=0.5,
        )

        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "UNSCORABLE",
                    "score": -1.0,
                }
            },
        }
        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        assert len(result) == 1
        assert result[0].mean == approx(0.5)

    def test_evaluate_generation_with_none_annotation_dict(self):
        """Test handling of None annotation dictionaries."""
        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "ACCEPT",
                    "score": 1.0,
                }
            },
            "annotator2": None,
        }
        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = self.metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        # Should only use annotator1
        assert len(result) == 1
        assert result[0].mean == approx(1.0)

    def test_evaluate_generation_multiple_nested_values(self):
        """Test handling of multiple nested score values per annotator."""
        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "ACCEPT",
                    "score": 1.0,
                }
            },
            "annotator2": {
                "accuracy": {
                    "classification": "CAUTION",
                    "score": 0.5,
                },
                "safety": {
                    "classification": "ACCEPT",
                    "score": 0.8,
                },
            },
        }
        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = self.metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        # Should average all scorable values: 1.0, 0.5, 0.8
        assert len(result) == 1
        assert result[0].mean == approx((1.0 + 0.5 + 0.8) / 3)

    def test_evaluate_generation_harm_level_0_scores(self):
        """Test harm_level 0 scoring (ACCEPT=1.0, CAUTION=0.0, REFUSE=0.0)."""
        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "ACCEPT",
                    "score": 1.0,
                }
            },
            "annotator2": {
                "accuracy": {
                    "classification": "CAUTION",
                    "score": 0.0,
                }
            },
            "annotator3": {
                "accuracy": {
                    "classification": "REFUSE",
                    "score": 0.0,
                }
            },
        }
        annotator_models_3 = {
            **self.annotator_models,
            "annotator3": AnnotatorModelInfo(
                model_name="meta/llama-2",
                model_deployment="meta/llama-2",
            ),
        }
        metric = MedSecurityLLMJuryMetric(
            metric_name=self.metric_name,
            scenario_name=self.scenario_name,
            annotator_models=annotator_models_3,
            default_score=0.0,
        )

        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        # Average of 1.0, 0.0, 0.0 = 0.333...
        assert len(result) == 1
        assert result[0].mean == approx((1.0 + 0.0 + 0.0) / 3)

    def test_evaluate_generation_harm_level_1_scores(self):
        """Test harm_level 1 scoring (ACCEPT=0.0, CAUTION=1.0, REFUSE=0.5)."""
        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "ACCEPT",
                    "score": 0.0,
                }
            },
            "annotator2": {
                "accuracy": {
                    "classification": "CAUTION",
                    "score": 1.0,
                }
            },
            "annotator3": {
                "accuracy": {
                    "classification": "REFUSE",
                    "score": 0.5,
                }
            },
        }
        annotator_models_3 = {
            **self.annotator_models,
            "annotator3": AnnotatorModelInfo(
                model_name="meta/llama-2",
                model_deployment="meta/llama-2",
            ),
        }
        metric = MedSecurityLLMJuryMetric(
            metric_name=self.metric_name,
            scenario_name=self.scenario_name,
            annotator_models=annotator_models_3,
            default_score=0.0,
        )

        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        # Average of 0.0, 1.0, 0.5 = 0.5
        assert len(result) == 1
        assert result[0].mean == approx(0.5)

    def test_evaluate_generation_metric_name_preservation(self):
        """Test that metric name is correctly preserved in output."""
        custom_metric_name = "custom_medsecurity_metric"
        metric = MedSecurityLLMJuryMetric(
            metric_name=custom_metric_name,
            scenario_name=self.scenario_name,
            annotator_models=self.annotator_models,
            default_score=0.0,
        )

        annotations = {
            "annotator1": {
                "accuracy": {
                    "classification": "ACCEPT",
                    "score": 1.0,
                }
            },
        }
        request_state = self._create_request_state(annotations)
        adapter_spec = MagicMock(spec=AdapterSpec)
        metric_service = MagicMock(spec=MetricService)

        result = metric.evaluate_generation(adapter_spec, request_state, metric_service, "")

        assert len(result) == 1
        assert result[0].name.name == custom_metric_name
