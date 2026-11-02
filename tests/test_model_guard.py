"""Proves the conftest guard actually blocks loading real model weights."""

import importlib

import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoPipelineForText2Image

GUARDED_CLASSES = [AutoTokenizer, AutoModelForCausalLM, AutoPipelineForText2Image]

# The names as the application code imports them.
APP_IMPORTS = [
    ("src.core.model_manager", "AutoTokenizer"),
    ("src.core.model_manager", "AutoModelForCausalLM"),
    ("src.media.image_generator", "AutoPipelineForText2Image"),
]


@pytest.mark.parametrize("model_class", GUARDED_CLASSES, ids=lambda c: c.__name__)
def test_from_pretrained_is_blocked(model_class):
    with pytest.raises(RuntimeError, match="Tests must not load real models"):
        model_class.from_pretrained("google/gemma-2-2b-it")


@pytest.mark.parametrize("module_name, class_name", APP_IMPORTS)
def test_from_pretrained_is_blocked_via_app_imports(module_name, class_name):
    model_class = getattr(importlib.import_module(module_name), class_name)
    with pytest.raises(RuntimeError, match="Tests must not load real models"):
        model_class.from_pretrained("any-model-id")
