"""Smoke test: src/core/config.py imports and has the expected defaults."""

import importlib
import logging

import src.core.config as config_module

# Defaults are read from environment variables at import time.
OVERRIDABLE_VARS = [
    "NUM_SCENES",
    "IMAGE_WIDTH",
    "IMAGE_HEIGHT",
    "IMAGE_INFERENCE_STEPS",
    "VIDEO_WIDTH",
    "VIDEO_HEIGHT",
    "VIDEO_FPS",
]


def test_config_defaults(monkeypatch):
    for name in OVERRIDABLE_VARS:
        monkeypatch.delenv(name, raising=False)
    # Teardown restores the original class, which the other modules still use
    monkeypatch.setattr(config_module, "Config", config_module.Config)
    Config = importlib.reload(config_module).Config

    assert Config.NUM_SCENES == 3
    assert (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT) == (512, 512)
    assert Config.IMAGE_INFERENCE_STEPS == 6
    assert (Config.VIDEO_WIDTH, Config.VIDEO_HEIGHT) == (512, 512)
    assert Config.VIDEO_FPS == 24


def test_summary_uses_measured_values(caplog):
    with caplog.at_level(logging.INFO, logger=config_module.logger.name):
        config_module.Config._log_config_summary()
    summary = caplog.text

    assert "3-5 minutes" not in summary
    assert "about 10 minutes per video on an RTX 2050 (4 GB)" in summary
    # VIDEO_DURATION_MIN/MAX are not enforced; a measured run gave 21.77 s
    assert "Duration: 15-20s" not in summary
