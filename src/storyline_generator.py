"""
Stage 1: Generate storyline from marketing prompt
Enhanced with input validation, retry logic, and better error handling
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
import torch
from src.config import Config
from src.model_manager import ModelManager

logger = logging.getLogger(__name__)


class StorylineGenerator:
    """Generate marketing storyline with comprehensive error handling"""

    def __init__(self):
        # Use shared model
        self.manager = ModelManager()
        self.model = self.manager.model
        self.tokenizer = self.manager.tokenizer
        self.use_gpu = self.manager.use_gpu
        logger.info("✅ StorylineGenerator ready (using shared model)")

    def _validate_input(self, marketing_prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Validate marketing prompt
        Returns (is_valid, error_message)
        """
        if not marketing_prompt or not marketing_prompt.strip():
            return False, "Marketing prompt cannot be empty"

        if len(marketing_prompt.strip()) < 10:
            return False, "Marketing prompt too short (minimum 10 characters)"

        if len(marketing_prompt) > 2000:
            return False, "Marketing prompt too long (maximum 2000 characters)"

        return True, None

    def _extract_text(self, value: Any) -> str:
        """Extract text from nested dict or return string"""
        if isinstance(value, dict):
            return value.get('text', value.get('description', str(value)))
        return str(value)

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from model output with multiple strategies
        """
        # Strategy 1: Find first complete JSON object
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Try to find JSON in code blocks
        try:
            code_block_start = '```json'
            code_block_end = '```'

            if code_block_start in text:
                start = text.find(code_block_start) + len(code_block_start)
                end = text.find(code_block_end, start)
                if end != -1:
                    json_str = text[start:end].strip()
                    return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Try to parse the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        return None

    def _validate_storyline(self, storyline: Dict[str, Any]) -> bool:
        """
        Validate generated storyline has required fields
        """
        required_fields = ['hook', 'problem', 'solution', 'benefit', 'cta']

        for field in required_fields:
            if field not in storyline:
                logger.warning(f"⚠️ Missing required field: {field}")
                return False

            # Check if field has content
            text = self._extract_text(storyline[field])
            if not text or len(text.strip()) < 5:
                logger.warning(f"⚠️ Field '{field}' has insufficient content")
                return False

        return True

    def generate(self, marketing_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generate storyline with retry logic and REALISTIC visual guidance
        """
        # Validate input
        is_valid, error_msg = self._validate_input(marketing_prompt)
        if not is_valid:
            logger.error(f"❌ Invalid input: {error_msg}")
            return None

        logger.info("📝 Generating storyline...")
        logger.info(f"📋 Prompt: {marketing_prompt[:100]}...")

        # Retry loop
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                logger.info(f"🔄 Attempt {attempt}/{Config.MAX_RETRIES}")

                # ENHANCED PROMPT with REALISTIC guidance
                prompt = f"""You are a marketing scriptwriter creating a video for THIS EXACT PRODUCT.

                PRODUCT DESCRIPTION:
                {marketing_prompt}

                CRITICAL RULES:
                1. Focus ONLY on the product features mentioned above
                2. The main benefit is: {marketing_prompt.split('.')[0] if '.' in marketing_prompt else marketing_prompt[:100]}
                3. Do NOT invent different scenarios - stick to the product
                4. Keep it simple and directly related to what the product does

                Create a 5-scene video storyline:

                Output ONLY valid JSON:
                {{
                  "hook": "Question about main product benefit, 12-20 words",
                  "problem": "Problem this product solves, 15-25 words",
                  "solution": "How this exact product works, 15-25 words",
                  "benefit": "Result customer gets, 15-25 words",
                  "cta": "Call to action to buy, 8-15 words",
                  "visual_theme": "modern cool",
                  "target_audience": "health-conscious adults",
                  "character_profile": "athletic person age 28-35 using this product",
                  "primary_setting": "location where this product is typically used",
                  "lighting_style": "bright natural lighting"
                }}

                EXAMPLE for "UV-C self-cleaning water bottle that kills bacteria":
                {{
                  "hook": "Worried about bacteria growing inside your water bottle?",
                  "problem": "Regular bottles harbor dangerous bacteria and need constant manual scrubbing to stay clean",
                  "solution": "This bottle uses built-in UV-C light to kill 99.9 percent of bacteria automatically in 60 seconds",
                  "benefit": "Drink pure bacteria-free water anytime without the hassle of washing by hand",
                  "cta": "Order your self-cleaning bottle now with special discount",
                  "visual_theme": "modern cool",
                  "target_audience": "fitness enthusiasts and health-conscious people",
                  "character_profile": "fit athletic person in gym clothes age 30",
                  "primary_setting": "modern gym or outdoor fitness area",
                  "lighting_style": "bright natural daylight"
                }}

                STAY FOCUSED on the actual product described above. Output ONLY JSON."""

                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                )

                if self.use_gpu:
                    inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=Config.MAX_NEW_TOKENS,
                        temperature=Config.TEMPERATURE,
                        top_p=Config.TOP_P,
                        top_k=Config.TOP_K,
                        repetition_penalty=Config.REPETITION_PENALTY,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                # Decode
                generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


                # Extract response (remove prompt)
                if prompt in generated_text:
                    response = generated_text.split(prompt)[-1].strip()
                else:
                    response = generated_text

                logger.info(f"📄 Raw response: {response[:200]}...")

                # Extract JSON
                storyline = self._extract_json_from_text(response)

                if storyline is None:
                    logger.warning(f"⚠️ Could not extract JSON on attempt {attempt}")
                    if attempt < Config.MAX_RETRIES:
                        time.sleep(Config.RETRY_DELAY)
                        continue
                    return None

                # Validate storyline
                if not self._validate_storyline(storyline):
                    logger.warning(f"⚠️ Invalid storyline structure on attempt {attempt}")
                    if attempt < Config.MAX_RETRIES:
                        time.sleep(Config.RETRY_DELAY)
                        continue
                    return None

                # Normalize text fields
                for key in ['hook', 'problem', 'solution', 'benefit', 'cta']:
                    if key in storyline:
                        storyline[key] = self._extract_text(storyline[key])

                # Add defaults for NEW optional fields with REALISTIC values
                if 'visual_theme' not in storyline:
                    storyline['visual_theme'] = 'professional blue'
                if 'target_audience' not in storyline:
                    storyline['target_audience'] = 'busy professionals'
                if 'character_profile' not in storyline:
                    storyline['character_profile'] = 'mid-30s professional woman with dark hair'
                if 'primary_setting' not in storyline:
                    storyline['primary_setting'] = 'modern office with natural light'
                if 'lighting_style' not in storyline:
                    storyline['lighting_style'] = 'natural morning light through windows'

                logger.info("✅ Storyline generated successfully")
                logger.info(f"📊 Hook: {storyline['hook'][:60]}...")
                logger.info(f"📊 Problem: {storyline['problem'][:60]}...")
                logger.info(f"📊 Solution: {storyline['solution'][:60]}...")
                logger.info(f"📸 Character: {storyline.get('character_profile', 'N/A')}")
                logger.info(f"📸 Setting: {storyline.get('primary_setting', 'N/A')}")
                logger.info(f"💡 Lighting: {storyline.get('lighting_style', 'N/A')}")

                return storyline

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"❌ GPU OOM on attempt {attempt}")
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                    if attempt < Config.MAX_RETRIES:
                        time.sleep(Config.RETRY_DELAY)
                        continue
                    return None
                else:
                    logger.error(f"❌ Runtime error: {str(e)}")
                    if attempt < Config.MAX_RETRIES:
                        time.sleep(Config.RETRY_DELAY)
                        continue
                    return None

            except Exception as e:
                logger.error(f"❌ Unexpected error on attempt {attempt}: {str(e)}")
                if attempt < Config.MAX_RETRIES:
                    time.sleep(Config.RETRY_DELAY)
                    continue
                return None

        logger.error("❌ Failed to generate storyline after all attempts")
        return None
