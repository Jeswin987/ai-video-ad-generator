"""
Scene 1's visual anchors must reach image generation (diagnosis item 1b).

SceneDescriptor used to drop character_anchor/setting_anchor/lighting_anchor
when normalizing scenes, so ImageGenerator guessed subject, setting and
lighting from keywords in the visual text instead.

Both classes are built without __init__, so no model is loaded.
"""

import pytest

import src.generation.scene_descriptor as scene_descriptor_module
from src.core.config import Config
from src.generation.scene_descriptor import SceneDescriptor
from src.generation.scene_normalizer import normalize_scenes
from src.media.image_generator import ImageGenerator

STORYLINE = {
    "hook": "The nets come up empty again.",
    "problem": "Old charts miss where the fish have moved.",
    "solution": "TideSense maps the shoals in real time.",
    "benefit": "Full nets before the sun is high.",
    "cta": "Download TideSense today.",
    "character_profile": "elderly fisherman with a grey beard",
    "primary_setting": "small stone harbour at dawn",
    "lighting_style": "golden hour light",
}

ANCHORS = {
    "character_anchor": "elderly fisherman with a grey beard and yellow oilskin coat",
    "setting_anchor": "small stone harbour at dawn with wooden boats",
    "lighting_anchor": "golden hour light from the east",
}

# Contains keywords ("woman", "desk", "office") that the fallback matching reacts to.
KEYWORD_VISUAL = "Woman sitting at desk in an office, looking at a laptop screen"


# ------------------------
# SceneDescriptor
# ------------------------

@pytest.fixture
def descriptor(monkeypatch):
    d = object.__new__(SceneDescriptor)
    d.visual_anchors = {
        "product_name": None,
        "character_description": "",
        "setting_description": "",
        "lighting_style": "",
        "color_palette": "",
    }
    d.previous_scenes = []

    def fake_generate_single_scene(prompt, scene_number, max_retries=2):
        scene = {
            "scene_number": scene_number,
            "duration": 5,
            "visual": f"Scene {scene_number}: the fisherman checks the screen on deck",
            "voiceover": f"Scene {scene_number} voiceover with enough words in it",
            "camera": "medium shot",
        }
        if scene_number == 1:
            scene.update(ANCHORS)
        return scene

    d._generate_single_scene = fake_generate_single_scene
    monkeypatch.setattr(Config, "NUM_SCENES", 3)
    monkeypatch.setattr(scene_descriptor_module.time, "sleep", lambda s: None)
    return d


def test_anchors_survive_scene_generation_and_normalization(descriptor):
    scenes = normalize_scenes(descriptor.generate_scenes(STORYLINE), STORYLINE)

    for key, value in ANCHORS.items():
        assert scenes[0][key] == value


# ------------------------
# ImageGenerator
# ------------------------

@pytest.fixture
def image_gen():
    gen = object.__new__(ImageGenerator)
    # Instance copy, so the class-level dict is not changed by the test
    gen._visual_context = {
        "product_name": None,
        "main_subject": None,
        "main_subject_details": None,
        "setting": None,
        "lighting_style": None,
        "color_palette": None,
        "camera_style": None,
        "clothing_style": None,
        "age_range": None,
    }
    return gen


def _first_scene_prompt(gen, scene):
    positive, _ = gen._build_cinematic_prompt(
        scene=scene, visual_theme="warm", scene_number=1, total_scenes=3
    )
    return positive


def test_anchors_win_over_keywords(image_gen):
    scene = {"visual_description": KEYWORD_VISUAL, **ANCHORS}
    prompt = _first_scene_prompt(image_gen, scene)

    for value in ANCHORS.values():
        assert value in prompt
    for keyword_value in ("young professional woman", "modern office workspace",
                          "soft professional studio lighting"):
        assert keyword_value not in prompt


def test_keywords_still_used_without_anchors(image_gen):
    prompt = _first_scene_prompt(image_gen, {"visual_description": KEYWORD_VISUAL})

    assert "young professional woman" in prompt
    assert "modern office workspace" in prompt
    assert "soft professional studio lighting" in prompt


def test_empty_anchor_falls_back_to_keywords_for_that_field_only(image_gen):
    scene = {
        "visual_description": KEYWORD_VISUAL,
        "character_anchor": ANCHORS["character_anchor"],
        "setting_anchor": "",
    }
    prompt = _first_scene_prompt(image_gen, scene)

    assert ANCHORS["character_anchor"] in prompt
    assert "young professional woman" not in prompt
    assert "modern office workspace" in prompt
    assert "soft professional studio lighting" in prompt
