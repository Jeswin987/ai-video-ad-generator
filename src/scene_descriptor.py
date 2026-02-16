"""
Enhanced Scene Descriptor with Sequential Generation and Visual Continuity
Generates scenes ONE AT A TIME for better small model performance
FIXED: tokenizer decode and safety checks
"""

import json
import logging
import time
import re
from typing import Dict, Any, List, Optional
import torch

from src.config import Config
from src.model_manager import ModelManager

logger = logging.getLogger(__name__)


class SceneDescriptor:
    """Generate scene descriptions with visual continuity tracking"""

    def __init__(self):
        # Reuse shared model
        self.manager = ModelManager()
        self.model = self.manager.model
        self.tokenizer = self.manager.tokenizer
        self.use_gpu = self.manager.use_gpu

        # Visual anchors established in scene 1
        self.visual_anchors = {
            "product_name": None,
            "character_description": "",
            "setting_description": "",
            "lighting_style": "",
            "color_palette": ""
        }

        # Progressive context
        self.previous_scenes = []
        logger.info("✅ SceneDescriptor ready with visual continuity")

    # ------------------------
    # Helpers
    # ------------------------

    def _extract_visual_elements(self, storyline: Dict[str, Any]) -> None:
        """Extract core visual elements from storyline for consistency"""
        solution = str(storyline.get("solution", "")).lower()
        benefit = str(storyline.get("benefit", "")).lower()
        combined_text = f"{solution} {benefit}"

        # Product name
        product_match = re.search(
            r"[A-Z][a-z]+(?:[A-Z][a-z]+)*", str(storyline.get("solution", ""))
        )
        if product_match:
            self.visual_anchors["product_name"] = product_match.group()

        # Use NEW character_profile if available
        if "character_profile" in storyline:
            self.visual_anchors["character_description"] = storyline["character_profile"]
        else:
            # Fallback to old detection
            if any(k in combined_text for k in ["woman", "female", "her"]):
                char_base = "young professional woman"
            elif any(k in combined_text for k in ["man", "male", "his"]):
                char_base = "young professional man"
            else:
                char_base = "young professional"
            self.visual_anchors["character_description"] = char_base

        # Use NEW primary_setting if available
        if "primary_setting" in storyline:
            self.visual_anchors["setting_description"] = storyline["primary_setting"]
        else:
            # Fallback to old detection
            if any(k in combined_text for k in ["office", "work"]):
                setting_base = "modern office"
            elif any(k in combined_text for k in ["gym", "fitness"]):
                setting_base = "fitness center"
            elif any(k in combined_text for k in ["home", "kitchen"]):
                setting_base = "contemporary home"
            else:
                setting_base = "professional setting"
            self.visual_anchors["setting_description"] = setting_base

        # Use NEW lighting_style if available
        if "lighting_style" in storyline:
            self.visual_anchors["lighting_style"] = storyline["lighting_style"]

        self.visual_anchors["color_palette"] = storyline.get("visual_theme", "professional blue")

        logger.info(
            f"📊 Visual Context: Character={self.visual_anchors['character_description']}, "
            f"Setting={self.visual_anchors['setting_description']}"
        )

    def build_establishing_prompt(self, storyline: Dict[str, Any]) -> str:
        """Build prompt for FIRST scene with REALISTIC visual descriptions"""
        hook = str(storyline.get('hook', ''))
        visual_theme = storyline.get('visual_theme', 'professional blue')

        # Use NEW fields from storyline if available
        character_hint = storyline.get('character_profile', 'mid-30s professional person')
        setting_hint = storyline.get('primary_setting', 'modern office')
        lighting_hint = storyline.get('lighting_style', 'natural morning light')

        prompt = f"""Generate scene 1 for a marketing video.

Story Hook: {hook}
Visual Theme: {visual_theme}
Character Type: {character_hint}
Setting Type: {setting_hint}
Lighting: {lighting_hint}

IMPORTANT: Describe visuals like a PHOTOGRAPHER, not an artist.

Instructions:
1. Describe the character like a real person (age 25-35, specific hair color, clothing details)
2. Describe the setting like a real location (modern office with glass windows, wooden desk, plants)
3. Describe lighting like a photographer (natural window light from left, soft shadows, golden hour)
4. Use concrete details, not abstract words

Output ONLY this JSON:
{{
  "scene_number": 1,
  "duration": 5,
  "character_anchor": "mid-30s woman with shoulder-length brown hair, wearing navy blue blazer and white shirt, natural makeup",
  "setting_anchor": "modern office with floor-to-ceiling windows, oak desk, green plants on shelf, white walls",
  "lighting_anchor": "bright natural daylight from left side, soft shadows, morning light at 10am",
  "visual": "Woman sitting at desk looking frustrated at laptop screen, papers scattered around, coffee cup nearby",
  "camera": "medium shot from slightly above, static camera",
  "voiceover": "{hook[:80]}"
}}

Be SPECIFIC. Use real-world details like actual colors, materials, times of day."""

        return prompt

    def build_continuation_prompt(self, storyline: Dict[str, Any], scene_number: int) -> str:
        """Build prompt for scenes 2-5 with continuity"""

        # FIXED: Safety check for previous_scenes
        if not self.previous_scenes or len(self.previous_scenes) == 0:
            # Fallback if no previous scene exists
            character_anchor = self.visual_anchors.get('character_description', 'professional person')
            setting_anchor = self.visual_anchors.get('setting_description', 'modern office')
            lighting_anchor = self.visual_anchors.get('lighting_style', 'natural light')
        else:
            first_scene = self.previous_scenes[0]
            character_anchor = first_scene.get('character_anchor', self.visual_anchors.get('character_description', ''))
            setting_anchor = first_scene.get('setting_anchor', self.visual_anchors.get('setting_description', ''))
            lighting_anchor = first_scene.get('lighting_anchor', self.visual_anchors.get('lighting_style', ''))

        scene_goals = {
            2: "Show the problem the character faces",
            3: "Introduce the product as the solution",
            4: "Show the product working successfully",
            5: "Show happy outcome with call-to-action"
        }
        goal = scene_goals.get(scene_number, "Continue the story")

        story_context = {
            2: storyline.get('problem', ''),
            3: storyline.get('solution', ''),
            4: storyline.get('benefit', ''),
            5: storyline.get('cta', '')
        }.get(scene_number, '')

        prompt = f"""Generate scene {scene_number} of 5 for marketing video.

CRITICAL - Keep EXACT same character and setting from scene 1:
- Character: {character_anchor}
- Setting: {setting_anchor}
- Lighting: {lighting_anchor}

Scene {scene_number} goal: {goal}
Message: {story_context}

Instructions:
1. Use the EXACT SAME character (same hair, same clothes, same age)
2. Use the EXACT SAME setting (same room, same furniture)
3. Keep SIMILAR lighting (same time of day, same light direction)
4. Describe what happens in this scene clearly and simply
5. Use photographer language (camera angles, lighting, composition)

Output ONLY this JSON:
{{
  "scene_number": {scene_number},
  "duration": 5,
  "visual": "Clear description of what character does, camera sees the same person and place from scene 1, in 20-25 words",
  "camera": "Camera angle and movement in 5-10 words",
  "voiceover": "What character says in 15-20 words",
  "continuity_note": "How this connects to previous scene in 10 words"
}}

Use REAL details. Describe like you're directing a photographer."""

        return prompt

    def _generate_single_scene(self, prompt: str, scene_number: int, max_retries: int = 2) -> Optional[Dict]:
        """Generate a single scene with retries"""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"  🔄 Scene {scene_number} attempt {attempt}/{max_retries}")

                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1536  # Reduced for single scene
                )

                if self.use_gpu:
                    inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

                # Generate with tighter parameters for small model
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=350,  # Enough for one scene
                        temperature=0.4,  # Lower for consistency
                        top_p=0.85,
                        top_k=40,
                        repetition_penalty=1.15,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # FIXED: Use batch_decode instead of decode
                generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                # Extract response
                response = (
                    generated_text.split(prompt)[-1].strip()
                    if prompt in generated_text
                    else generated_text
                )

                logger.info(f"  📄 Response: {len(response)} chars")

                # Extract JSON
                scene = self._extract_json_from_text(response)

                if scene and isinstance(scene, dict):
                    # Ensure scene number
                    scene["scene_number"] = scene_number

                    # Validate basic fields
                    if self._validate_scene(scene, scene_number):
                        logger.info(f"  ✅ Scene {scene_number} generated")
                        return scene
                    else:
                        logger.warning(f"  ⚠️ Scene {scene_number} validation failed")
                else:
                    logger.warning(f"  ⚠️ Scene {scene_number} JSON extraction failed")

                if attempt < max_retries:
                    time.sleep(1)

            except Exception as e:
                logger.error(f"  ❌ Scene {scene_number} error: {str(e)}")
                if attempt < max_retries:
                    time.sleep(1)

        return None

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text"""
        # Strategy 1: Find JSON object
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Try entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        return None

    def _validate_scene(self, scene: Dict[str, Any], scene_num: int) -> bool:
        """Validate scene structure"""
        # Scene 1 requires anchors
        if scene_num == 1:
            required = ["scene_number", "duration", "character_anchor", "setting_anchor", "visual", "voiceover"]
        else:
            required = ["scene_number", "duration", "visual", "voiceover"]

        for field in required:
            if field not in scene:
                return False

            # Check content length for text fields
            if field in ["visual", "voiceover", "character_anchor", "setting_anchor"]:
                val = str(scene[field]).strip()
                if len(val) < 15:
                    return False

        return True

    def _create_fallback_scene(self, scene_num: int, storyline: Dict) -> Dict:
        """Create fallback scene using ACTUAL storyline content"""
        # Map scene numbers to actual storyline content
        scene_content = {
            1: {
                'visual': f"Person looking at {storyline.get('problem', 'a problem situation')}",
                'voiceover': storyline.get('hook', 'Introducing our solution')
            },
            2: {
                'visual': f"Showing the problem: {storyline.get('problem', 'common challenge')}",
                'voiceover': storyline.get('problem', 'Facing a common challenge')
            },
            3: {
                'visual': f"Product showcase: {storyline.get('solution', 'the solution')}",
                'voiceover': storyline.get('solution', 'Discover the product')
            },
            4: {
                'visual': f"Benefits demonstration: {storyline.get('benefit', 'the benefits')}",
                'voiceover': storyline.get('benefit', 'Experience the benefits')
            },
            5: {
                'visual': f"Call to action: {storyline.get('cta', 'take action')}",
                'voiceover': storyline.get('cta', 'Get started today')
            }
        }

        content = scene_content.get(scene_num, scene_content[1])

        fallback_scene = {
            "scene_number": scene_num,
            "duration": 5,
            "visual": content['visual'],
            "voiceover": content['voiceover'],
            "camera": "medium shot, static",
            "visual_description": content['visual'],
            "voiceover_script": content['voiceover']
        }

        # Add anchors for scene 1
        if scene_num == 1:
            fallback_scene["character_anchor"] = self.visual_anchors.get("character_description", "professional person")
            fallback_scene["setting_anchor"] = self.visual_anchors.get("setting_description", "modern setting")
            fallback_scene["lighting_anchor"] = self.visual_anchors.get("lighting_style", "natural lighting")

        return fallback_scene

    def _update_anchors_from_scene_1(self, scene: Dict):
        """Extract and store visual anchors from first scene"""
        self.visual_anchors["character_description"] = scene.get("character_anchor", "")
        self.visual_anchors["setting_description"] = scene.get("setting_anchor", "")
        self.visual_anchors["lighting_style"] = scene.get("lighting_anchor", "")
        logger.info(f"  🎯 Anchors set: {self.visual_anchors['character_description'][:50]}...")

    # ------------------------
    # Main Generation
    # ------------------------

    def generate_scenes(self, storyline: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Generate scenes SEQUENTIALLY with visual continuity"""
        logger.info("🎬 Generating scenes sequentially with continuity...")

        # Extract visual hints
        self._extract_visual_elements(storyline)

        # Validate storyline
        hook = str(storyline.get("hook", ""))
        problem = str(storyline.get("problem", ""))
        solution = str(storyline.get("solution", ""))

        if not all([hook, problem, solution]):
            logger.error("❌ Incomplete storyline")
            return None

        total_scenes = Config.NUM_SCENES
        self.previous_scenes = []

        # Generate each scene sequentially
        for scene_num in range(1, total_scenes + 1):
            logger.info(f"🎬 Generating scene {scene_num}/{total_scenes}...")

            # Build prompt based on scene number
            if scene_num == 1:
                prompt = self.build_establishing_prompt(storyline)
            else:
                prompt = self.build_continuation_prompt(storyline, scene_num)

            # Generate this scene
            scene = self._generate_single_scene(prompt, scene_num)

            if scene:
                self.previous_scenes.append(scene)

                # Extract anchors from first scene
                if scene_num == 1:
                    self._update_anchors_from_scene_1(scene)
            else:
                # Use fallback
                logger.warning(f"⚠️ Using fallback for scene {scene_num}")
                fallback = self._create_fallback_scene(scene_num, storyline)
                self.previous_scenes.append(fallback)

                # If scene 1 failed, update anchors from fallback
                if scene_num == 1:
                    self._update_anchors_from_scene_1(fallback)

            # Brief pause between generations
            if scene_num < total_scenes:
                time.sleep(0.5)

        # Normalize scene format for downstream compatibility
        normalized_scenes = []
        for scene in self.previous_scenes:
            normalized = {
                "scene_number": scene.get("scene_number", 0),
                "duration": scene.get("duration", 5),
                "visual_description": scene.get("visual", "Scene visual"),
                "voiceover_script": scene.get("voiceover", "Scene audio"),
                "camera_motion": scene.get("camera", "medium shot"),
                "text_overlay": scene.get("text_overlay", "")
            }
            normalized_scenes.append(normalized)

        logger.info(f"✅ Generated {len(normalized_scenes)} scenes with continuity")
        for s in normalized_scenes:
            logger.info(f"  Scene {s['scene_number']}: {s['voiceover_script'][:50]}...")

        return normalized_scenes
