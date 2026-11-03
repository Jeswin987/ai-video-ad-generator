"""Server run arguments in app.py, checked without starting the server."""

import importlib

import pytest

import app
import src.core.config as config_module


@pytest.fixture
def load_config(monkeypatch):
    """Reload config.py so APP_HOST is read again; put the original Config back afterwards."""
    # Teardown restores the original class, which the other modules still use
    monkeypatch.setattr(config_module, "Config", config_module.Config)

    def _load():
        return importlib.reload(config_module).Config

    return _load


def test_host_defaults_to_localhost(monkeypatch, load_config):
    monkeypatch.delenv("APP_HOST", raising=False)

    assert app.get_run_options(load_config())["host"] == "127.0.0.1"


def test_host_comes_from_app_host(monkeypatch, load_config):
    monkeypatch.setenv("APP_HOST", "0.0.0.0")

    assert app.get_run_options(load_config())["host"] == "0.0.0.0"


def test_debug_is_off(monkeypatch, load_config):
    monkeypatch.delenv("APP_HOST", raising=False)

    options = app.get_run_options(load_config())

    assert options["debug"] is False
    assert options["port"] == 5000
