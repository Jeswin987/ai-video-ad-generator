"""
The startup VRAM check must not flag a 4 GB card as below 4 GB.

An RTX 2050 reports total_memory just under 4 GiB, so `< 4` fired the
"Less than 4GB" error on the card the project targets.

torch.cuda is faked: no real CUDA calls.
"""

import logging
from types import SimpleNamespace

import pytest

import src.core.config as config_module
from src.core.config import Config

GIB = 1024 ** 3


@pytest.fixture
def fake_gpu(monkeypatch):
    def install(total_gib):
        monkeypatch.setattr(Config, "DEVICE", "cuda")
        monkeypatch.setattr(config_module.torch.cuda, "get_device_name", lambda i: "Fake GPU")
        monkeypatch.setattr(
            config_module.torch.cuda,
            "get_device_properties",
            lambda i: SimpleNamespace(total_memory=int(total_gib * GIB)),
        )
    return install


def test_card_just_under_4_gib_gets_4gb_tier(fake_gpu, caplog):
    fake_gpu(3.996)
    with caplog.at_level(logging.INFO, logger=config_module.logger.name):
        Config._validate_device()

    assert "Less than 4GB VRAM" not in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "4GB VRAM - using aggressive optimization" in caplog.text


def test_3_gib_card_still_gets_error(fake_gpu, caplog):
    fake_gpu(3.0)
    with caplog.at_level(logging.INFO, logger=config_module.logger.name):
        Config._validate_device()

    assert "Less than 4GB VRAM - GPU mode not recommended" in caplog.text
    assert "4GB VRAM - using aggressive optimization" not in caplog.text
