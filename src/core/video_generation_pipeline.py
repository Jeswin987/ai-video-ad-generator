"""
Complete Video Generation Pipeline with AI Image Integration
Orchestrates: Text Generation → Image Generation → Video Creation → Audio Merge
Optimized VRAM management: Never loads text and image models together
ENHANCED: Scene normalization, duration sync, and optional upscaling
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import time

from src.core.config import Config
from src.generation.storyline_generator import StorylineGenerator
from src.generation.scene_descriptor import SceneDescriptor
from src.media.image_generator import ImageGenerator
from src.media.voiceover_generator import VoiceoverGenerator
from src.media.video_generator_with_images import VideoGeneratorWithImages
from src.media.audio_merger import AudioMerger
from src.media.media_merger import MediaMerger
from src.generation.scene_normalizer import normalize_scenes
from src.media.image_upscaler import ImageUpscaler

logger = logging.getLogger(__name__)


class VideoGenerationPipeline:
    """Complete pipeline with AI image generation and enhancements"""

    def __init__(self):
        logger.info("🚀 Initializing VideoGenerationPipeline with AI images...")

        # Initialize text generators (they share the model)
        self.storyline_gen = StorylineGenerator()
        self.scene_gen = SceneDescriptor()

        # Image generator will be loaded on-demand
        self.image_gen = None

        # Initialize other generators
        self.voiceover_gen = VoiceoverGenerator()
        self.audio_merger = AudioMerger()
        self.video_gen = None  # Will be initialized after images
        self.media_merger = MediaMerger()

        # Optional upscaler
        self.upscaler = ImageUpscaler() if Config.ENABLE_UPSCALING else None

        logger.info("✅ Pipeline ready")

    def generate_video(
            self,
            prompt: str,
            visual_theme: str = "professional blue",
            output_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate complete video with AI images

        Pipeline:
        1. Generate storyline (Gemma-2-2B)
        2. Generate scenes (Gemma-2-2B)
        2.5. Normalize scenes
        3. Unload text model
        4. Generate images (SDXL-Turbo)
        4.5. Optional upscaling
        5. Unload image model
        6. Generate voiceovers with duration sync
        7. Create videos from images
        8. Merge continuous audio
        9. Final video assembly
        """
        try:
            if output_dir is None:
                output_dir = Config.OUTPUT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("=" * 70)
            logger.info("🎬 STARTING VIDEO GENERATION PIPELINE")
            logger.info("=" * 70)
            logger.info(f"📝 Prompt: {prompt[:100]}...")
            logger.info(f"🎨 Theme: {visual_theme}")
            logger.info(f"🖼️ Image Generation: {'Enabled' if Config.USE_IMAGE_GENERATION else 'Disabled'}")
            logger.info(f"🔍 Upscaling: {'Enabled' if Config.ENABLE_UPSCALING else 'Disabled'}")
            logger.info("=" * 70)

            # ================================================================
            # STAGE 1: Generate storyline
            # ================================================================
            logger.info("\n📖 STAGE 1: Generating storyline...")
            storyline = self.storyline_gen.generate(prompt)
            if not storyline:
                logger.error("❌ Storyline generation failed")
                return None

            # ================================================================
            # STAGE 2: Generate scenes
            # ================================================================
            logger.info("\n🎬 STAGE 2: Generating scene descriptions...")
            scenes = self.scene_gen.generate_scenes(storyline)
            if not scenes:
                logger.error("❌ Scene generation failed")
                return None
            logger.info(f"✅ Generated {len(scenes)} scenes")

            # ================================================================
            # STAGE 2.5: Normalize scenes
            # ================================================================
            logger.info("\n🔧 STAGE 2.5: Normalizing scene schema...")
            scenes = normalize_scenes(scenes, storyline)
            logger.info("✅ Scene normalization complete")

            # Validation checkpoint
            logger.info("🔍 Validating normalized scenes...")
            for idx, scene in enumerate(scenes, start=1):
                assert "visual_description" in scene, f"Scene {idx} missing visual_description"
                assert "voiceover_script" in scene, f"Scene {idx} missing voiceover_script"
                assert "text_overlay" in scene, f"Scene {idx} missing text_overlay"
                assert isinstance(scene["visual_description"], str), f"Scene {idx} visual_description not string"
                assert isinstance(scene["voiceover_script"], str), f"Scene {idx} voiceover_script not string"
                assert isinstance(scene["text_overlay"], str), f"Scene {idx} text_overlay not string"
            logger.info("✅ All scenes validated")

            # ================================================================
            # STAGE 3: Generate AI Images
            # ================================================================
            scene_images = {}
            if Config.USE_IMAGE_GENERATION:
                logger.info("\n🎨 STAGE 3: Generating AI images...")
                logger.info("⚠️ Unloading text model to free VRAM...")

                # Unload text model completely
                self._unload_text_models()

                # Load image generator
                logger.info("📥 Loading SDXL-Turbo...")
                self.image_gen = ImageGenerator()

                # Generate ONE image per scene
                images_dir = output_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                scene_images = self.image_gen.generate_batch(
                    scenes=scenes,
                    output_dir=images_dir,
                    visual_theme=visual_theme
                )

                if len(scene_images) < len(scenes):
                    logger.warning(f"⚠️ Only {len(scene_images)}/{len(scenes)} images generated")

                # Unload image model
                logger.info("🗑️ Unloading SDXL-Turbo to free VRAM...")
                self.image_gen.unload_model()
                self.image_gen = None
                torch.cuda.empty_cache()

                logger.info(f"✅ Generated {len(scene_images)} AI images")
            else:
                logger.info("\n🎨 STAGE 3: Skipping image generation (using theme colors)")

            # ================================================================
            # STAGE 3.5: Optional Image Upscaling
            # ================================================================
            if Config.ENABLE_UPSCALING and scene_images and Config.UPSCALE_BEFORE_VIDEO:
                logger.info("\n🔍 STAGE 3.5: Upscaling images for enhanced quality...")

                upscaled_dir = output_dir / "images_upscaled"
                upscaled_dir.mkdir(parents=True, exist_ok=True)

                scene_images = self.upscaler.batch_upscale(
                    image_paths=scene_images,
                    output_dir=upscaled_dir
                )

                upscaled_width, upscaled_height = self.upscaler.get_upscaled_dimensions(
                    Config.IMAGE_WIDTH,
                    Config.IMAGE_HEIGHT
                )

                logger.info(f"✅ Images upscaled to {Config.UPSCALE_FACTOR}x resolution")
                logger.info(f"📐 Upscaled dimensions: {upscaled_width}x{upscaled_height}")

            # ================================================================
            # STAGE 4: Generate voiceovers WITH DURATION SYNC
            # ================================================================
            logger.info("\n🎤 STAGE 4: Generating voiceovers with duration sync...")
            audio_dir = output_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            audio_files = []
            audio_durations = []

            for idx, scene in enumerate(scenes):
                scene_num = scene.get('scene_number', idx + 1)
                audio_filename = f"audio_scene_{scene_num}_{int(time.time() * 1000)}.mp3"

                # Generate voiceover with duration sync
                logger.info(f"🎤 Scene {scene_num}: Generating voiceover...")

                if Config.DURATION_SYNC_ENABLED:
                    # Use duration sync method
                    target_duration = scene.get('target_duration')  # Optional target
                    audio_path, measured_duration = self.voiceover_gen.generate_with_duration_sync(
                        scene=scene,
                        output_filename=audio_filename,
                        target_duration=target_duration if Config.ALLOW_DURATION_ADJUSTMENT else None,
                        tolerance=Config.DURATION_SYNC_TOLERANCE
                    )
                else:
                    # Legacy method
                    audio_path = self.voiceover_gen.generate(scene, audio_filename)
                    measured_duration = self.voiceover_gen.get_audio_duration(audio_path) if audio_path else None

                if not audio_path:
                    logger.error(f"❌ Failed to generate audio for Scene {scene_num}")
                    return None

                # Update scene duration to match measured audio
                if measured_duration:
                    old_duration = scene.get('duration', 5.0)

                    # Clamp to min/max bounds
                    measured_duration = max(
                        Config.MIN_SCENE_DURATION,
                        min(Config.MAX_SCENE_DURATION, measured_duration)
                    )

                    scene['duration'] = measured_duration
                    audio_durations.append(measured_duration)

                    # Log if significant change
                    if abs(measured_duration - old_duration) > Config.DURATION_SYNC_TOLERANCE:
                        logger.info(
                            f"🔄 Scene {scene_num}: Duration synced "
                            f"{old_duration:.2f}s → {measured_duration:.2f}s"
                        )
                    else:
                        logger.info(f"✅ Scene {scene_num}: Duration {measured_duration:.2f}s")
                else:
                    # Fallback to default
                    fallback_duration = scene.get('duration', 5.0)
                    audio_durations.append(fallback_duration)
                    logger.warning(
                        f"⚠️ Scene {scene_num}: Could not measure duration, "
                        f"using {fallback_duration:.2f}s"
                    )

                audio_files.append(audio_path)

                # Small delay to avoid timestamp collisions
                time.sleep(0.01)

            if len(audio_files) != len(scenes):
                logger.error(f"❌ Audio generation incomplete: {len(audio_files)}/{len(scenes)}")
                return None

            # Log duration summary
            total_duration = sum(audio_durations)
            self._log_duration_summary(scenes, total_duration)

            logger.info(f"✅ Generated {len(audio_files)} voiceovers with synced durations")

            # ================================================================
            # STAGE 5: Merge continuous audio
            # ================================================================
            logger.info("\n🎵 STAGE 5: Merging continuous audio track...")
            merged_audio_path = output_dir / "merged_audio.mp3"
            try:
                merged_audio = self.audio_merger.merge_scene_audio(
                    audio_files=audio_files,
                    output_path=merged_audio_path,
                    crossfade_ms=Config.AUDIO_CROSSFADE_MS
                )
                logger.info(f"✅ Continuous audio created: {merged_audio.name}")
            except Exception as e:
                logger.warning(f"⚠️ Audio merge failed: {str(e)}")
                logger.info("📌 Will use individual scene audio as fallback")
                merged_audio = None

            # ================================================================
            # STAGE 6: Generate videos
            # ================================================================
            logger.info("\n🎥 STAGE 6: Creating videos...")
            videos_dir = output_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)

            # Initialize video generator
            self.video_gen = VideoGeneratorWithImages()

            video_files = []
            for i, scene in enumerate(scenes):
                scene_num = scene['scene_number']

                # Get AI image for this scene (if available)
                scene_image = scene_images.get(scene_num) if scene_images else None

                # Generate video
                video_path = self.video_gen.generate_scene_video(
                    scene=scene,
                    output_dir=videos_dir,
                    visual_theme=visual_theme,
                    scene_image_path=scene_image
                )

                if video_path:
                    video_files.append(video_path)

            if len(video_files) != len(scenes):
                logger.error(f"❌ Video generation incomplete: {len(video_files)}/{len(scenes)}")
                return None

            logger.info(f"✅ Generated {len(video_files)} scene videos")

            # ================================================================
            # STAGE 7: Merge final video
            # ================================================================
            logger.info("\n🎬 STAGE 7: Assembling final video...")

            if merged_audio:
                # If multiple videos, concatenate them first
                if len(video_files) > 1:
                    logger.info("🔗 Concatenating scene videos...")
                    final_video = self.media_merger.concatenate_videos(
                        video_paths=video_files,
                        output_filename="concatenated_video.mp4"
                    )
                else:
                    final_video = video_files[0]

                # Then add continuous audio
                logger.info("🎵 Adding continuous audio to final video...")
                final_video = self.media_merger.merge_scene(
                    video_path=final_video,
                    audio_path=merged_audio,
                    output_filename="final_video_with_audio.mp4"
                )
            else:
                # Fallback: merge individual audio per scene
                logger.info("📌 Merging individual audio per scene (fallback)...")
                merged_videos = []
                for idx, (video, audio) in enumerate(zip(video_files, audio_files)):
                    merged = self.media_merger.merge_scene(
                        video_path=video,
                        audio_path=audio,
                        output_filename=f"merged_scene_{idx + 1}.mp4"
                    )
                    if merged:
                        merged_videos.append(merged)

                # Concatenate all merged videos
                if len(merged_videos) > 1:
                    logger.info("🔗 Concatenating merged scene videos...")
                    final_video = self.media_merger.concatenate_videos(
                        video_paths=merged_videos,
                        output_filename="final_video.mp4"
                    )
                else:
                    final_video = merged_videos[0] if merged_videos else None

            # ================================================================
            # FINAL SUMMARY
            # ================================================================
            logger.info("=" * 70)
            logger.info("✅ VIDEO GENERATION COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"📁 Final video: {final_video}")
            logger.info(f"📊 Scenes: {len(scenes)}")
            logger.info(f"🖼️ AI Images: {len(scene_images) if scene_images else 0}")
            logger.info(f"🔍 Upscaled: {'Yes' if Config.ENABLE_UPSCALING and scene_images else 'No'}")
            logger.info(f"🎵 Continuous Audio: {'Yes' if merged_audio else 'No'}")
            logger.info(f"⏱️ Total Duration: {total_duration:.2f}s ({total_duration / 60:.1f} min)")
            logger.info("=" * 70)

            return final_video

        except Exception as e:
            logger.error(f"❌ Pipeline error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

        finally:
            # Cleanup
            self._cleanup()

    def _log_duration_summary(self, scenes: List[Dict[str, Any]], total_duration: float):
        """Log duration summary after sync"""
        logger.info("\n" + "=" * 70)
        logger.info("⏱️  DURATION SUMMARY")
        logger.info("=" * 70)

        for scene in scenes:
            scene_num = scene.get('scene_number', 0)
            duration = scene.get('duration', 0.0)
            logger.info(f"  Scene {scene_num}: {duration:.2f}s")

        avg = total_duration / len(scenes) if scenes else 0

        logger.info("-" * 70)
        logger.info(f"  Total: {total_duration:.2f}s ({total_duration / 60:.1f} minutes)")
        logger.info(f"  Average: {avg:.2f}s per scene")
        logger.info("=" * 70 + "\n")

    def _unload_text_models(self):
        """Unload text generation models to free VRAM"""
        try:
            # Access the shared model manager
            from src.core.model_manager import ModelManager
            manager = ModelManager()

            # Clear model
            if hasattr(manager, 'model'):
                del manager.model
            if hasattr(manager, 'tokenizer'):
                del manager.tokenizer

            # Reset initialization flag
            ModelManager._initialized = False
            ModelManager._model = None
            ModelManager._tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("✅ Text models unloaded")
        except Exception as e:
            logger.warning(f"⚠️ Text model unload warning: {str(e)}")

    def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.image_gen:
                self.image_gen.unload_model()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("🧹 Cleanup complete")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {str(e)}")
