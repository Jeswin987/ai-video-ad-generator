"""
Shared test fixtures.

Guards against loading real model weights: tests must never load Gemma or
SDXL-Turbo, use the GPU, or download anything.
"""

import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import AutoPipelineForText2Image

import src.core.config as config_module


@pytest.fixture(autouse=True)
def config_class_not_replaced():
    """Fail if a test leaves src.core.config.Config as a different class (e.g. after a reload).

    Other modules keep the class they imported, so patches on a replaced
    Config would silently miss them. block_real_model_loading requests this
    fixture before monkeypatch, so this check runs after monkeypatch has
    undone its patches.
    """
    original = config_module.Config
    yield
    if config_module.Config is not original:
        pytest.fail("src.core.config.Config was replaced and not restored (config.py reloaded?)")

# Classes the code uses to load models (src/core/model_manager.py and
# src/media/image_generator.py). Patched on the classes themselves, so the
# guard applies however they are imported.
GUARDED_CLASSES = [AutoTokenizer, AutoModelForCausalLM, AutoPipelineForText2Image]


def _blocked_from_pretrained(cls, *args, **kwargs):
    raise RuntimeError(
        f"Tests must not load real models: {cls.__name__}.from_pretrained was called"
    )


@pytest.fixture(autouse=True)
def block_real_model_loading(config_class_not_replaced, monkeypatch):
    for model_class in GUARDED_CLASSES:
        monkeypatch.setattr(model_class, "from_pretrained", classmethod(_blocked_from_pretrained))
