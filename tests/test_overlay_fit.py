"""
Overlay text must fit inside the video frame.

Run B drew each overlay as one line wider than the 512 px frame, cut off
on both sides. fit_overlay_text wraps into at most 2 lines and shrinks the
font when needed.

Uses Pillow's built-in font, so no system fonts are needed.
"""

import pytest
from PIL import ImageFont

from src.media.video_generator_with_images import (
    MIN_OVERLAY_FONT_SIZE,
    fit_overlay_text,
)

MAX_WIDTH = 448  # 512 px frame minus a margin on each side

SHORT = "Try NoteFlow free"
TWO_LINES = "Tired of losing the thread in your notes?"
TOO_LONG_FOR_TWO_LINES = "Download Noteflow Free Today Start Revising Smarter"


@pytest.fixture
def font():
    return ImageFont.load_default(size=40)


def assert_fits(lines, font):
    assert 1 <= len(lines) <= 2
    for line in lines:
        assert font.getlength(line) <= MAX_WIDTH, line


def test_short_text_stays_on_one_line(font):
    lines, fitted = fit_overlay_text(SHORT, font, MAX_WIDTH)

    assert lines == [SHORT]
    assert fitted.size == 40


def test_long_text_wraps_to_two_lines_at_full_size(font):
    assert font.getlength(TWO_LINES) > MAX_WIDTH  # would overflow as one line

    lines, fitted = fit_overlay_text(TWO_LINES, font, MAX_WIDTH)

    assert len(lines) == 2
    assert fitted.size == 40
    assert " ".join(lines) == TWO_LINES  # split at word boundaries, nothing lost
    assert_fits(lines, fitted)


def test_text_too_long_for_two_lines_gets_smaller_font(font):
    lines, fitted = fit_overlay_text(TOO_LONG_FOR_TWO_LINES, font, MAX_WIDTH)

    assert MIN_OVERLAY_FONT_SIZE <= fitted.size < 40
    assert " ".join(lines) == TOO_LONG_FOR_TWO_LINES
    assert_fits(lines, fitted)


@pytest.mark.parametrize("text", [
    SHORT,
    TWO_LINES,
    TOO_LONG_FOR_TWO_LINES,
    "Every semester the same story repeats itself, with...",
    "Supercalifragilisticexpialidocious" * 3,  # one word wider than the frame
    " ".join(["revision"] * 60),  # too long even at the minimum size
])
def test_no_line_is_ever_wider_than_max_width(font, text):
    lines, fitted = fit_overlay_text(text, font, MAX_WIDTH)

    assert fitted.size >= MIN_OVERLAY_FONT_SIZE
    assert_fits(lines, fitted)
