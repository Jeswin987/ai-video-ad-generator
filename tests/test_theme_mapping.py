"""
Theme names from the web UI must reach the image prompt's palette.

The UI used to send names like "Modern Cool", which matched none of the
theme keys in image_generator.py, so every choice got the default palette.

No model is loaded: ImageGenerator is built without __init__, and the API
test replaces the job's task.
"""

import logging
import threading

import pytest

import app as app_module
from src.media.image_generator import DEFAULT_THEME, THEME_COLORS, ImageGenerator, resolve_theme


@pytest.mark.parametrize("name, key", [
    ("Modern Cool", "cool"),
    ("Warm Professional", "warm"),
])
def test_ui_display_names_map_to_their_theme(name, key):
    assert resolve_theme(name) == key


@pytest.mark.parametrize("key", list(THEME_COLORS))
def test_recognised_keys_map_to_themselves(key):
    assert resolve_theme(key) == key


@pytest.mark.parametrize("name, key", [
    ("MODERN COOL", "cool"),
    ("modern cool", "cool"),
    ("Cool", "cool"),
    ("Professional Blue", "professional blue"),
])
def test_matching_ignores_case(name, key):
    assert resolve_theme(name) == key


@pytest.mark.parametrize("name", ["Bold Energetic", "Elegant Minimal", "neon"])
def test_unknown_name_gets_default_and_warning(name, caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_theme(name) == DEFAULT_THEME

    assert name in caplog.text


def test_missing_theme_gets_default_without_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_theme(None) == DEFAULT_THEME

    assert caplog.text == ""


def test_default_is_a_recognised_theme():
    assert DEFAULT_THEME in THEME_COLORS


def test_display_name_reaches_the_image_prompt():
    gen = object.__new__(ImageGenerator)
    gen._visual_context = {key: None for key in ImageGenerator._visual_context}

    positive, _ = gen._build_cinematic_prompt(
        scene={"visual_description": "a desk"}, visual_theme="Modern Cool",
        scene_number=1, total_scenes=3
    )

    assert THEME_COLORS["cool"] in positive


def test_api_job_gets_the_resolved_theme(monkeypatch, tmp_path):
    task_themes = []
    task_ran = threading.Event()

    def fake_task(job_id, prompt, visual_theme, output_dir):
        task_themes.append(visual_theme)
        task_ran.set()

    monkeypatch.setattr(app_module, "generate_video_task", fake_task)
    monkeypatch.setattr(app_module.Config, "OUTPUT_DIR", tmp_path)

    response = app_module.app.test_client().post(
        "/api/generate-video", json={"prompt": "A water bottle", "theme": "Warm Professional"}
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    assert app_module.jobs[job_id]["visual_theme"] == "warm"
    assert task_ran.wait(timeout=5)
    assert task_themes == ["warm"]
