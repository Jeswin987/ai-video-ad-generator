"""
Each scene gets its own overlay text (bug B9).

Every scene used to show the same overlay, derived from the storyline CTA.
Now scenes show a phrase from their own storyline beat (scene 1 = hook,
2 = problem, 3 = solution, 4 = benefit), and the last scene keeps the
CTA-based overlay.
"""

from src.generation.scene_normalizer import derive_text_overlay, normalize_scenes

STORYLINE = {
    "hook": "Tired of losing the thread in your notes?",
    "problem": "Ideas get buried under old lecture slides.",
    "solution": "NoteFlow links every idea to its source.",
    "benefit": "Revise in half the time.",
    "cta": "Download NoteFlow free today and start revising smarter",
}


def make_scenes(count):
    return [
        {
            "scene_number": n,
            "visual_description": f"Scene {n} visual",
            "voiceover_script": f"Scene {n} voiceover",
            "text_overlay": "",
        }
        for n in range(1, count + 1)
    ]


def overlays(storyline, count=3):
    return [s["text_overlay"] for s in normalize_scenes(make_scenes(count), storyline)]


def test_three_scenes_get_three_different_overlays():
    result = overlays(STORYLINE)

    assert len(set(result)) == 3
    # Beat text is shown as written: no stopword removal, no title-casing
    assert result[0] == "Tired of losing the thread in your notes?"
    assert result[1] == "Ideas get buried under old lecture slides."


def test_last_scene_keeps_the_cta_overlay():
    for count in (3, 5):
        result = overlays(STORYLINE, count)
        assert result[-1] == derive_text_overlay(f"Scene {count} voiceover", STORYLINE)


def test_long_beat_is_cut_at_a_word_boundary():
    storyline = dict(
        STORYLINE,
        hook="Every semester the same story repeats itself, with notes scattered everywhere",
    )

    assert overlays(storyline)[0] == "Every semester the same story repeats itself, with..."


def test_beat_of_exactly_eight_words_is_not_cut():
    storyline = dict(STORYLINE, hook="One two three four five six seven eight")

    assert overlays(storyline)[0] == "One two three four five six seven eight"


def test_missing_beat_falls_back_to_cta_overlay():
    storyline = {k: v for k, v in STORYLINE.items() if k != "problem"}

    result = overlays(storyline)

    assert result[1] == derive_text_overlay("Scene 2 voiceover", storyline)
    assert result[0] == STORYLINE["hook"]
