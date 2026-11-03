"""
The pipeline must really unload Gemma before image generation (bug B1).

Uses fake model and tokenizer objects: no real weights, no CUDA.
"""

import gc
import logging
import weakref

import pytest
import torch

import src.core.config as config_module
import src.core.video_generation_pipeline as pipeline_module
from src.core.model_manager import ModelManager
from src.core.video_generation_pipeline import VideoGenerationPipeline


class FakeModel:
    pass


class FakeTokenizer:
    pass


class FakeComponent:
    """Stands in for the pipeline's media helpers, which check for ffmpeg/gTTS."""


@pytest.fixture
def load_calls(monkeypatch):
    """Fresh ModelManager singleton whose load step installs fakes and counts calls."""
    monkeypatch.setattr(ModelManager, "_instance", None)
    monkeypatch.setattr(ModelManager, "_model", None)
    monkeypatch.setattr(ModelManager, "_tokenizer", None)
    monkeypatch.setattr(ModelManager, "_initialized", False)
    monkeypatch.setattr(ModelManager, "_device", None)

    calls = []

    def fake_load(self):
        calls.append(1)
        ModelManager._model = FakeModel()
        ModelManager._tokenizer = FakeTokenizer()
        ModelManager._device = "cpu"

    monkeypatch.setattr(ModelManager, "_load_model", fake_load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # The patch must reach the Config class the pipeline actually reads
    assert pipeline_module.Config is config_module.Config
    monkeypatch.setattr(config_module.Config, "ENABLE_UPSCALING", False)
    for name in ("VoiceoverGenerator", "AudioMerger", "MediaMerger"):
        monkeypatch.setattr(pipeline_module, name, FakeComponent)
    return calls


def test_unload_releases_every_reference_to_the_model(load_calls):
    pipeline = VideoGenerationPipeline()
    model_ref = weakref.ref(ModelManager._model)
    tokenizer_ref = weakref.ref(ModelManager._tokenizer)

    pipeline._unload_text_models()
    gc.collect()

    assert model_ref() is None
    assert tokenizer_ref() is None


def test_unload_logs_no_warning(load_calls, caplog):
    pipeline = VideoGenerationPipeline()

    with caplog.at_level(logging.WARNING):
        pipeline._unload_text_models()

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_next_job_loads_gemma_again(load_calls):
    first = VideoGenerationPipeline()
    first._unload_text_models()

    VideoGenerationPipeline()

    assert len(load_calls) == 2
