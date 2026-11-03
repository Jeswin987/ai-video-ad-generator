"""
Scene prompts must be built from the storyline, not from a fixed example.

Gemma-2-2B tends to copy the prompt's JSON example, so a hard-coded office
scene in the example ends up in every video (diagnosis item 1a).

Builds SceneDescriptor without __init__, so no model is loaded.
"""

import pytest

from src.generation.scene_descriptor import SceneDescriptor

STORYLINE = {
    "hook": "The nets come up empty again.",
    "problem": "Old charts miss where the fish have moved.",
    "solution": "TideSense maps the shoals in real time.",
    "benefit": "Full nets before the sun is high.",
    "cta": "Download TideSense today.",
    "visual_theme": "warm",
    "character_profile": "elderly fisherman with a grey beard and yellow oilskin coat",
    "primary_setting": "small stone harbour at dawn",
    "lighting_style": "golden hour light",
}

# Content of the old fixed example and of the office-only instruction hints.
HARD_CODED_EXAMPLE_STRINGS = [
    "mid-30s woman",
    "shoulder-length brown hair",
    "navy blue blazer",
    "natural makeup",
    "floor-to-ceiling windows",
    "oak desk",
    "green plants on shelf",
    "morning light at 10am",
    "Woman sitting at desk",
    "frustrated at laptop",
    "modern office",
    "glass windows",
    "wooden desk",
]


@pytest.fixture
def descriptor():
    d = object.__new__(SceneDescriptor)
    d.visual_anchors = {
        "product_name": None,
        "character_description": "",
        "setting_description": "",
        "lighting_style": "",
        "color_palette": "",
    }
    d.previous_scenes = []
    return d


def _assert_storyline_only(prompt):
    for hint in ("elderly fisherman", "harbour at dawn", "golden hour light"):
        assert hint in prompt
    leaked = [s for s in HARD_CODED_EXAMPLE_STRINGS if s.lower() in prompt.lower()]
    assert leaked == []


def test_establishing_prompt_example_comes_from_storyline(descriptor):
    _assert_storyline_only(descriptor.build_establishing_prompt(STORYLINE))


@pytest.mark.parametrize("scene_number", [2, 3, 4, 5])
def test_continuation_prompt_uses_storyline_anchors(descriptor, scene_number):
    descriptor._extract_visual_elements(STORYLINE)
    _assert_storyline_only(descriptor.build_continuation_prompt(STORYLINE, scene_number))
