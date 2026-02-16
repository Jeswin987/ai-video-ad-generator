"""
Easing Functions for Smooth Ken Burns Motion
Provides various easing curves for natural-looking camera movements
"""

import math
from typing import Callable


def ease_linear(t: float) -> float:
    """Linear progression (no easing)"""
    return t


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in (slow start, accelerate)"""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out (fast start, decelerate)"""
    return t * (2 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out (smooth both ends)"""
    return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in (very slow start)"""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out (very slow end)"""
    return 1 + (t - 1) ** 3


def ease_in_out_cubic(t: float) -> float:
    """
    Cubic ease-in-out (recommended default)
    Smooth acceleration and deceleration
    """
    return 4 * t * t * t if t < 0.5 else 1 + (t - 1) * (2 * (t - 2)) ** 2


def ease_in_sine(t: float) -> float:
    """Sine ease-in (gentle start)"""
    return 1 - math.cos(t * math.pi / 2)


def ease_out_sine(t: float) -> float:
    """Sine ease-out (gentle end)"""
    return math.sin(t * math.pi / 2)


def ease_in_out_sine(t: float) -> float:
    """Sine ease-in-out (very smooth)"""
    return -(math.cos(math.pi * t) - 1) / 2


def ease_out_back(t: float) -> float:
    """
    Back ease-out (slight overshoot for emphasis)
    Good for product reveals
    """
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def get_easing_function(name: str) -> Callable[[float], float]:
    """
    Get easing function by name

    Args:
        name: Name of easing function

    Returns:
        Easing function that maps [0,1] → [0,1]
    """
    easing_map = {
        "linear": ease_linear,
        "ease_in_quad": ease_in_quad,
        "ease_out_quad": ease_out_quad,
        "ease_in_out_quad": ease_in_out_quad,
        "ease_in_cubic": ease_in_cubic,
        "ease_out_cubic": ease_out_cubic,
        "ease_in_out_cubic": ease_in_out_cubic,
        "ease_in_sine": ease_in_sine,
        "ease_out_sine": ease_out_sine,
        "ease_in_out_sine": ease_in_out_sine,
        "ease_out_back": ease_out_back,
    }

    return easing_map.get(name, ease_in_out_cubic)
