"""
Enhanced Image Generator with Visual Continuity & Realistic Prompts
Generates professional marketing images with scene-to-scene consistency
OPTIMIZED FOR 4GB VRAM (RTX 2050)
FIXED: Auto-detects FP16 variant availability
"""

import logging
import torch
import gc
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from PIL import Image
from diffusers import AutoPipelineForText2Image

from src.config import Config

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images using SDXL-Turbo with 4GB VRAM optimization"""

    _instance = None
    _pipe = None
    _initialized = False

    # Track visual elements for consistency
    _visual_context = {
        'product_name': None,
        'main_subject': None,
        'main_subject_details': None,
        'setting': None,
        'lighting_style': None,
        'color_palette': None,
        'camera_style': None,
        'clothing_style': None,
        'age_range': None
    }

    # Simplified negative prompt for SDXL-Turbo (1-step model)
    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, low quality, distorted, deformed, "
        "watermark, text, multiple people, cartoon, 3d render"
    )

    def __new__(cls):
        """Singleton pattern - only one instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize image generator (only once)"""
        if not ImageGenerator._initialized:
            logger.info("🎨 Initializing Enhanced ImageGenerator (4GB mode)...")
            self._load_pipeline()
            ImageGenerator._initialized = True

    def _load_pipeline(self):
        """
        Load SDXL-Turbo with AGGRESSIVE 4GB VRAM optimization
        FIXED: Auto-detect variant availability (fp16 or fp32)
        """
        model_path = Config.MODELS_DIR / "sdxl-turbo"

        try:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"SDXL-Turbo model not found at: {model_path}\n"
                    f"Please run: python download_sdxl_turbo.py"
                )

            # CRITICAL: Clear cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            logger.info(f"📂 Loading SDXL-Turbo (4GB optimized) from: {model_path}")

            # FIXED: Check if fp16 variant exists
            fp16_available = (model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists()

            if not fp16_available:
                logger.warning("⚠️  FP16 variant not found - loading FP32 and converting to FP16")
            else:
                logger.info("✅ FP16 variant detected")

            # FIXED: Load with automatic variant detection
            load_kwargs = {
                "torch_dtype": torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                "use_safetensors": True,
                "local_files_only": True,
                "low_cpu_mem_usage": True
            }

            # Only add variant parameter if fp16 files exist
            if fp16_available and Config.DEVICE == "cuda":
                load_kwargs["variant"] = "fp16"

            ImageGenerator._pipe = AutoPipelineForText2Image.from_pretrained(
                str(model_path),
                **load_kwargs
            )

            if Config.DEVICE == "cuda":
                # FIXED: Enable sequential CPU offload BEFORE moving to GPU
                # This keeps only active layers on GPU (uses ~2-3GB instead of 7GB)
                ImageGenerator._pipe.enable_sequential_cpu_offload()

                # Enable attention slicing (reduces VRAM during generation)
                ImageGenerator._pipe.enable_attention_slicing(1)

                # Enable VAE slicing if available (reduces VRAM for decode)
                if hasattr(ImageGenerator._pipe, 'enable_vae_slicing'):
                    ImageGenerator._pipe.enable_vae_slicing()

                # Enable VAE tiling for large images (optional but helpful)
                if hasattr(ImageGenerator._pipe, 'enable_vae_tiling'):
                    ImageGenerator._pipe.enable_vae_tiling()

                logger.info("✅ SDXL-Turbo loaded with sequential CPU offload (4GB mode)")
                logger.info("   - Only active layers stay on GPU")
                logger.info("   - Attention slicing enabled")
                logger.info("   - VAE optimizations enabled")
            else:
                logger.info("✅ SDXL-Turbo loaded on CPU")

            # Check memory usage
            if Config.DEVICE == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                logger.info(f"💾 Initial VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        except Exception as e:
            logger.error(f"❌ Failed to load SDXL-Turbo: {str(e)}")
            raise

    def _extract_visual_context(self, scene: dict, is_first_scene: bool = False) -> None:
        """
        Extract and store visual context from scene for consistency
        ENHANCED: More detailed extraction for better continuity
        """
        visual_desc = scene.get('visual_description', '').lower()

        if is_first_scene or not self._visual_context['main_subject']:
            # Extract key visual elements from first scene

            # Subject identification (ENHANCED)
            if 'woman' in visual_desc or 'female' in visual_desc:
                self._visual_context['main_subject'] = 'young professional woman'
                # Try to extract more details
                if 'blonde' in visual_desc:
                    self._visual_context['main_subject_details'] = 'blonde hair'
                elif 'brunette' in visual_desc or 'dark hair' in visual_desc:
                    self._visual_context['main_subject_details'] = 'dark hair'

                if 'business' in visual_desc or 'suit' in visual_desc:
                    self._visual_context['clothing_style'] = 'professional business attire'
                elif 'casual' in visual_desc:
                    self._visual_context['clothing_style'] = 'smart casual attire'

            elif 'man' in visual_desc or 'male' in visual_desc:
                self._visual_context['main_subject'] = 'young professional man'
                if 'beard' in visual_desc:
                    self._visual_context['main_subject_details'] = 'with beard'

                if 'business' in visual_desc or 'suit' in visual_desc:
                    self._visual_context['clothing_style'] = 'professional business attire'
                elif 'casual' in visual_desc:
                    self._visual_context['clothing_style'] = 'smart casual attire'

            elif 'person' in visual_desc or 'people' in visual_desc:
                self._visual_context['main_subject'] = 'young professional'

            # Age range detection (NEW)
            if 'young' in visual_desc or '20s' in visual_desc or '30s' in visual_desc:
                self._visual_context['age_range'] = 'mid-20s to early 30s'

            # Extract setting
            if 'office' in visual_desc or 'desk' in visual_desc or 'workspace' in visual_desc:
                self._visual_context['setting'] = 'modern office workspace'
            elif 'gym' in visual_desc or 'fitness' in visual_desc or 'workout' in visual_desc:
                self._visual_context['setting'] = 'modern fitness center'
            elif 'home' in visual_desc or 'kitchen' in visual_desc or 'living' in visual_desc:
                self._visual_context['setting'] = 'contemporary modern home interior'
            elif 'outdoor' in visual_desc or 'park' in visual_desc or 'nature' in visual_desc:
                self._visual_context['setting'] = 'outdoor natural setting'
            elif 'cafe' in visual_desc or 'coffee' in visual_desc:
                self._visual_context['setting'] = 'modern cafe interior'
            else:
                self._visual_context['setting'] = 'clean professional minimalist setting'

            # Extract lighting
            if 'morning' in visual_desc or 'sunrise' in visual_desc:
                self._visual_context['lighting_style'] = 'warm golden hour morning light'
            elif 'evening' in visual_desc or 'sunset' in visual_desc:
                self._visual_context['lighting_style'] = 'warm sunset golden hour lighting'
            elif 'bright' in visual_desc or 'day' in visual_desc or 'daylight' in visual_desc:
                self._visual_context['lighting_style'] = 'bright natural daylight streaming through windows'
            elif 'dramatic' in visual_desc:
                self._visual_context['lighting_style'] = 'dramatic cinematic lighting with highlights'
            else:
                self._visual_context['lighting_style'] = 'soft professional studio lighting'

            logger.info(f"📌 Visual Context Anchors Set:")
            logger.info(f"   Subject: {self._visual_context['main_subject']}")
            if self._visual_context['main_subject_details']:
                logger.info(f"   Details: {self._visual_context['main_subject_details']}")
            logger.info(f"   Setting: {self._visual_context['setting']}")
            logger.info(f"   Lighting: {self._visual_context['lighting_style']}")

    def _build_consistency_elements(self) -> str:
        """
        Build consistency string from stored visual context
        NEW: Ensures same subject, setting, and lighting across scenes
        """
        elements = []

        # Subject consistency (CRITICAL for character continuity)
        if self._visual_context['main_subject']:
            subject_str = f"the SAME {self._visual_context['main_subject']}"

            if self._visual_context['main_subject_details']:
                subject_str += f" {self._visual_context['main_subject_details']}"

            if self._visual_context['age_range']:
                subject_str += f", {self._visual_context['age_range']}"

            if self._visual_context['clothing_style']:
                subject_str += f", wearing {self._visual_context['clothing_style']}"

            elements.append(f"featuring {subject_str}")

        # Setting consistency
        if self._visual_context['setting']:
            elements.append(f"in the SAME {self._visual_context['setting']}")

        # Lighting consistency
        if self._visual_context['lighting_style']:
            elements.append(f"with consistent {self._visual_context['lighting_style']}")

        return ", ".join(elements) if elements else ""

    def _build_cinematic_prompt(
            self,
            scene: dict,
            visual_theme: str,
            scene_number: int,
            total_scenes: int
    ) -> Tuple[str, str]:
        """
        Build cinematic prompt optimized for SDXL-Turbo
        FIXED: Shorter prompts work better with turbo models
        RETURNS: (positive_prompt, negative_prompt)
        """
        visual_desc = scene.get('visual_description', '')

        # Extract context for first scene
        if scene_number == 1:
            self._extract_visual_context(scene, is_first_scene=True)

        # Theme-based colors
        theme_colors = {
            'professional blue': 'blue and white corporate colors',
            'green': 'green and natural earth tones',
            'purple': 'purple and gold luxury colors',
            'warm': 'warm orange and yellow tones',
            'cool': 'cool blue minimalist tones'
        }

        color_palette = theme_colors.get(visual_theme.lower(), 'professional blue tones')

        # Camera angle progression
        camera_angles = [
            'medium shot',
            'close-up',
            'product hero shot',
            'action shot',
            'wide final shot'
        ]

        camera_angle = camera_angles[min(scene_number - 1, len(camera_angles) - 1)]

        # Build consistency elements
        consistency_str = self._build_consistency_elements()

        # FIXED: Shorter, focused prompt for SDXL-Turbo (1-step model)
        enhanced_prompt = (
            f"professional photograph, {visual_desc}, {camera_angle}, "
            f"{color_palette}, {consistency_str}, "
            f"commercial photography, sharp focus, "
            f"cinematic lighting, photorealistic, 8k quality"
        )

        # Clean up prompt
        enhanced_prompt = " ".join(enhanced_prompt.split())

        # FIXED: Simpler negative prompt for turbo model
        negative_prompt = self.DEFAULT_NEGATIVE_PROMPT
        if scene_number > 1:
            negative_prompt += ", different person, inconsistent character"

        return enhanced_prompt, negative_prompt

    def _save_prompt_metadata(
            self,
            scene_number: int,
            positive_prompt: str,
            negative_prompt: str,
            output_dir: Path,
            visual_theme: str
    ) -> None:
        """
        Save prompt strings for auditability and tuning
        NEW: Creates JSON metadata file alongside image
        """
        try:
            metadata = {
                "scene_number": scene_number,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "visual_theme": visual_theme,
                "visual_context": dict(self._visual_context),
                "model": "stabilityai/sdxl-turbo",
                "inference_steps": Config.IMAGE_INFERENCE_STEPS,
                "guidance_scale": 0.0,  # FIXED: Always 0 for turbo
                "dimensions": f"{Config.IMAGE_WIDTH}x{Config.IMAGE_HEIGHT}"
            }

            metadata_path = output_dir / f"scene_{scene_number}_prompt.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"📝 Prompt metadata saved: {metadata_path.name}")

        except Exception as e:
            logger.warning(f"⚠️ Could not save prompt metadata: {str(e)}")

    def generate_scene_image(
            self,
            scene: dict,
            output_dir: Path,
            visual_theme: str = "professional blue",
            scene_number: int = 1,
            total_scenes: int = 5
    ) -> Optional[Path]:
        """
        Generate cinematic marketing image with visual continuity
        ENHANCED: Now with aggressive 4GB VRAM memory management
        """
        try:
            scene_num = scene.get('scene_number', scene_number)
            visual_desc = scene.get('visual_description', '')

            if not visual_desc:
                logger.warning(f"⚠️ Scene {scene_num} has no visual description")
                return None

            logger.info(f"🎨 Generating cinematic image for Scene {scene_num}...")

            # Build enhanced cinematic prompt
            positive_prompt, negative_prompt = self._build_cinematic_prompt(
                scene=scene,
                visual_theme=visual_theme,
                scene_number=scene_num,
                total_scenes=total_scenes
            )

            logger.info(f"📝 Prompt: {positive_prompt[:100]}...")

            # CRITICAL: Clear cache before generation
            if Config.DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            # Generate with SDXL-Turbo
            with torch.inference_mode():
                result = ImageGenerator._pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=Config.IMAGE_INFERENCE_STEPS,
                    guidance_scale=0.0,  # FIXED: Turbo models work best with 0.0
                    height=Config.IMAGE_HEIGHT,
                    width=Config.IMAGE_WIDTH,
                    num_images_per_prompt=1
                )

                image = result.images[0]

            # Save image
            output_path = output_dir / f"scene_{scene_num}_image.png"
            image.save(output_path, format='PNG', optimize=True)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Scene {scene_num} image saved: {output_path.name} ({file_size_mb:.2f}MB)")

            # Save prompt metadata
            self._save_prompt_metadata(
                scene_number=scene_num,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                output_dir=output_dir,
                visual_theme=visual_theme
            )

            # CRITICAL: Aggressive cleanup after generation
            if Config.DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            return output_path

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"❌ GPU OOM generating Scene {scene_num} image")
                if Config.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                return None
            else:
                logger.error(f"❌ Runtime error: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"❌ Image generation error for Scene {scene_num}: {str(e)}")
            return None

    def generate_batch(
            self,
            scenes: list,
            output_dir: Path,
            visual_theme: str = "professional blue"
    ) -> dict:
        """
        Generate consistent images for all scenes with visual continuity
        """
        logger.info(f"🎨 Generating {len(scenes)} cinematic images with visual continuity...")

        # Reset visual context for new batch
        self._visual_context = {
            'product_name': None,
            'main_subject': None,
            'main_subject_details': None,
            'setting': None,
            'lighting_style': None,
            'color_palette': None,
            'camera_style': None,
            'clothing_style': None,
            'age_range': None
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        scene_images = {}
        total_scenes = len(scenes)

        for idx, scene in enumerate(scenes, 1):
            scene_num = scene.get('scene_number', idx)

            image_path = self.generate_scene_image(
                scene=scene,
                output_dir=output_dir,
                visual_theme=visual_theme,
                scene_number=scene_num,
                total_scenes=total_scenes
            )

            if image_path:
                scene_images[scene_num] = image_path
                logger.info(f"  Progress: {idx}/{total_scenes} images generated")

                # ADDED: Memory check after each image
                if Config.DEVICE == "cuda":
                    allocated = torch.cuda.memory_allocated() / 1024 ** 3
                    logger.info(f"  💾 VRAM used: {allocated:.2f}GB")
            else:
                logger.warning(f"⚠️ Failed to generate image for Scene {scene_num}")

        success_rate = len(scene_images) / total_scenes * 100
        logger.info(f"✅ Generated {len(scene_images)}/{total_scenes} images ({success_rate:.0f}% success)")
        logger.info(
            f"📊 Visual Context: Subject={self._visual_context['main_subject']}, "
            f"Setting={self._visual_context['setting']}"
        )

        return scene_images

    def clear_cache(self):
        """Clear GPU cache to free VRAM"""
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 GPU cache cleared")

    def unload_model(self):
        """Unload model to free VRAM for other operations"""
        if ImageGenerator._pipe is not None:
            del ImageGenerator._pipe
            ImageGenerator._pipe = None
            ImageGenerator._initialized = False

            if Config.DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            logger.info("🗑️ Image generator unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded"""
        return ImageGenerator._pipe is not None

    def get_memory_stats(self) -> dict:
        """Get current VRAM usage"""
        if Config.DEVICE != "cuda":
            return {"device": "cpu", "memory": "N/A"}

        try:
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            return {
                "device": "cuda",
                "allocated_gb": f"{allocated:.2f}",
                "reserved_gb": f"{reserved:.2f}",
                "total_gb": f"{total:.2f}",
                "free_gb": f"{total - reserved:.2f}"
            }

        except Exception as e:
            logger.warning(f"⚠️ Could not get memory stats: {str(e)}")
            return {"device": "cuda", "error": str(e)}
