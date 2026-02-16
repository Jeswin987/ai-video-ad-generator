"""
Scene Normalization Layer
Standardizes scene schema for downstream modules
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Configuration
MIN_OVERLAY_WORDS = 5
MAX_OVERLAY_WORDS = 8

# Common stopwords to filter for cleaner overlays
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "so", "to", "for", "of", "in",
    "on", "at", "with", "your", "you", "we", "our", "us", "is", "are",
    "be", "it", "this", "that", "as", "by", "from", "will", "can"
}


def _clean_text(text: str) -> List[str]:
    """
    Clean text and return list of words

    Args:
        text: Input text

    Returns:
        List of cleaned words
    """
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r"[^A-Za-z0-9' ]+", " ", text)
    # Split and filter empty strings
    words = [w.strip() for w in text.split() if w.strip()]
    return words


def _create_overlay(words: List[str], target_length: int) -> str:
    """
    Create title-cased overlay from word list

    Args:
        words: List of words
        target_length: Target number of words

    Returns:
        Title-cased overlay string
    """
    selected = words[:target_length]
    return " ".join(selected).title()


def derive_text_overlay(voiceover: str, storyline: Dict[str, Any]) -> str:
    """
    Derive concise text overlay (5-8 words) from voiceover or storyline

    Priority:
    1. Storyline CTA
    2. Storyline headline/benefit
    3. First meaningful words from voiceover

    Args:
        voiceover: Scene voiceover script
        storyline: Complete storyline dictionary

    Returns:
        Concise text overlay (5-8 words, title-cased)
    """
    # Try storyline CTA first (best for marketing)
    cta = (storyline or {}).get("cta", "").strip()
    if cta:
        source = cta
    else:
        # Try storyline headline or benefit
        headline = (storyline or {}).get("headline", "").strip()
        benefit = (storyline or {}).get("benefit", "").strip()
        source = headline or benefit or voiceover.strip()

    if not source:
        return "Learn More"  # Safe fallback

    # Clean and extract words
    words = _clean_text(source)

    if not words:
        return "Learn More"

    # Filter stopwords for more impactful overlay
    filtered = [w for w in words if w.lower() not in STOPWORDS]

    # If filtering removes too much, use original words
    if len(filtered) < MIN_OVERLAY_WORDS:
        filtered = words

    # Target length between min and max
    target_len = max(MIN_OVERLAY_WORDS, min(MAX_OVERLAY_WORDS, len(filtered)))

    return _create_overlay(filtered, target_len)


def normalize_scene(scene: Dict[str, Any], storyline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single scene to standard schema

    Ensures all downstream modules receive:
    - visual_description (for ImageGenerator)
    - voiceover_script (for VoiceoverGenerator)
    - text_overlay (for VideoGeneratorWithImages)

    Args:
        scene: Raw scene dictionary from SceneDescriptor
        storyline: Complete storyline dictionary

    Returns:
        Normalized scene dictionary with standardized keys
    """
    normalized = dict(scene)  # Shallow copy to avoid mutations

    # Map visual → visual_description
    normalized["visual_description"] = (
            normalized.get("visual_description") or
            normalized.get("visual") or
            ""
    ).strip()

    # Map voiceover → voiceover_script
    normalized["voiceover_script"] = (
            normalized.get("voiceover_script") or
            normalized.get("voiceover") or
            ""
    ).strip()

    # Derive or validate text_overlay
    existing_overlay = normalized.get("text_overlay", "").strip()
    if not existing_overlay:
        normalized["text_overlay"] = derive_text_overlay(
            normalized["voiceover_script"],
            storyline
        )
    else:
        normalized["text_overlay"] = existing_overlay

    return normalized


def normalize_scenes(
        scenes: List[Dict[str, Any]],
        storyline: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Normalize all scenes and apply guardrails

    Args:
        scenes: List of raw scene dictionaries
        storyline: Complete storyline dictionary

    Returns:
        List of normalized scene dictionaries
    """
    normalized_scenes = []

    for idx, scene in enumerate(scenes, start=1):
        # Normalize schema
        norm_scene = normalize_scene(scene, storyline)

        # Apply safety guardrails for downstream modules

        # Ensure visual_description is never empty (ImageGenerator requirement)
        if not norm_scene["visual_description"]:
            logger.warning(
                f"⚠️ Scene {idx}: Empty visual_description, using fallback"
            )
            norm_scene["visual_description"] = (
                "Professional high-quality marketing product photography"
            )

        # Ensure voiceover_script is never empty (VoiceoverGenerator requirement)
        if not norm_scene["voiceover_script"]:
            logger.warning(
                f"⚠️ Scene {idx}: Empty voiceover_script, using fallback"
            )
            # Use minimal non-empty string to prevent TTS failure
            norm_scene["voiceover_script"] = "."

        # Ensure text_overlay is never empty (VideoGeneratorWithImages requirement)
        if not norm_scene["text_overlay"]:
            logger.warning(
                f"⚠️ Scene {idx}: Empty text_overlay, using fallback"
            )
            norm_scene["text_overlay"] = "Learn More"

        # Ensure scene_number is set
        if "scene_number" not in norm_scene:
            norm_scene["scene_number"] = idx

        normalized_scenes.append(norm_scene)

    logger.info(f"✅ Normalized {len(normalized_scenes)} scenes")

    # Log first scene for verification
    if normalized_scenes:
        first = normalized_scenes[0]
        logger.debug(
            f"📋 Scene 1 preview:\n"
            f"  - visual_description: {first['visual_description'][:60]}...\n"
            f"  - voiceover_script: {first['voiceover_script'][:60]}...\n"
            f"  - text_overlay: {first['text_overlay']}"
        )

    return normalized_scenes
