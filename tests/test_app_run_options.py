"""Server run arguments in app.py, checked without starting the server."""

import importlib

import pytest

import app
import src.core.config as config_module


@pytest.fixture
def load_config(monkeypatch):
    """Reload config.py so APP_HOST is read again; restore a clean Config afterwards."""

    def _load():
        return importlib.reload(config_module).Config

    yield _load
    monkeypatch.delenv("APP_HOST", raising=False)
    importlib.reload(config_module)


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
