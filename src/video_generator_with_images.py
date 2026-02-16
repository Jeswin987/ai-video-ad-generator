"""
Video Generator WITH AI Images and Ken Burns Effect
Creates videos from AI-generated images instead of colored backgrounds
ENHANCED: Fixed resolution, audio-aware motion with easing, scene-specific presets
FIXED: Ken Burns clamping logic to prevent dimension mismatches
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, Union

from src.config import Config

logger = logging.getLogger(__name__)


class VideoGeneratorWithImages:
    """Generate videos using AI images with enhanced Ken Burns effect"""

    def __init__(self):
        # FIXED: Lock to Config dimensions - never change
        self.width = Config.VIDEO_WIDTH
        self.height = Config.VIDEO_HEIGHT
        self.fps = Config.VIDEO_FPS
        self._current_motion_preset = None  # For scene-specific motion

        logger.info(f"🎥 VideoGeneratorWithImages ready (AI-powered)")
        logger.info(f"📐 Fixed resolution: {self.width}x{self.height} @ {self.fps}fps")
        logger.info(f"🎬 Ken Burns: {'Enabled' if Config.KEN_BURNS_ENABLED else 'Disabled'}")
        if Config.KEN_BURNS_ENABLED:
            logger.info(f"   Easing: {Config.KEN_BURNS_EASING}")
            logger.info(f"   Audio-aware: {'Yes' if Config.KEN_BURNS_AUDIO_AWARE else 'No'}")

    def generate_scene_video(
            self,
            scene: dict,
            output_dir: Path,
            visual_theme: str,
            scene_image_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate video for ONE scene using AI image
        WITH SCENE-SPECIFIC MOTION
        """
        try:
            scene_num = scene.get('scene_number', 1)
            duration = float(scene.get('duration', 5.0))
            text_overlay = scene.get('text_overlay', '')

            # Get motion preset for this scene type
            motion_preset = self._get_motion_preset(scene)
            self._current_motion_preset = motion_preset

            logger.info(f"🎥 Generating video for Scene {scene_num}...")
            logger.info(f"   Duration: {duration}s")
            logger.info(f"   Resolution: {self.width}x{self.height}")
            logger.info(f"   Motion: {motion_preset['description']}")

            if scene_image_path and scene_image_path.exists():
                logger.info(f"   Using AI image: {scene_image_path.name}")
                return self._generate_with_ai_image(
                    scene_num=scene_num,
                    duration=duration,
                    image_path=scene_image_path,
                    text_overlay=text_overlay,
                    output_dir=output_dir
                )
            else:
                logger.warning(f"   No AI image, using theme: {visual_theme}")
                return self._generate_with_theme(
                    scene_num=scene_num,
                    duration=duration,
                    theme=visual_theme,
                    text_overlay=text_overlay,
                    output_dir=output_dir
                )

        except Exception as e:
            logger.error(f"❌ Video generation error for Scene {scene_num}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _get_motion_preset(self, scene: dict) -> dict:
        """
        Get Ken Burns motion preset based on scene type

        Returns:
            Dict with zoom_factor, pan_intensity, easing, description
        """
        scene_num = scene.get('scene_number', 1)

        # Scene-specific motion profiles for storytelling
        presets = {
            1: {  # Scene 1: Establishing shot
                'zoom_factor': 1.08,
                'pan_intensity': 0.5,
                'easing': 'ease_in_out_sine',
                'description': 'Gentle establishing motion'
            },
            2: {  # Scene 2: Problem reveal
                'zoom_factor': 1.12,
                'pan_intensity': 0.7,
                'easing': 'ease_in_quad',
                'description': 'Gradual tension build'
            },
            3: {  # Scene 3: Solution/Product
                'zoom_factor': 1.15,
                'pan_intensity': 0.4,
                'easing': 'ease_out_cubic',
                'description': 'Confident product reveal'
            },
            4: {  # Scene 4: Benefits demonstration
                'zoom_factor': 1.10,
                'pan_intensity': 0.8,
                'easing': 'ease_in_out_cubic',
                'description': 'Dynamic demonstration'
            },
            5: {  # Scene 5: Call to action
                'zoom_factor': 1.06,
                'pan_intensity': 0.3,
                'easing': 'ease_out_sine',
                'description': 'Calm, confident close'
            }
        }

        return presets.get(scene_num, presets[1])

    def _load_and_prepare_image(self, image_path: Path) -> Image.Image:
        """
        Load image and prepare it for video generation
        FIXED: Always resize/crop to Config dimensions
        """
        # Load image
        pil_image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = pil_image.size
        logger.info(f"   Original image size: {orig_width}x{orig_height}")

        # Target dimensions from Config
        target_width = self.width
        target_height = self.height
        target_aspect = target_width / target_height
        orig_aspect = orig_width / orig_height

        # Strategy: Smart crop/resize to fill target dimensions
        if abs(orig_aspect - target_aspect) < 0.01:
            # Aspect ratios match - simple resize
            prepared = pil_image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            logger.info(f"   Resized to: {target_width}x{target_height}")

        elif orig_aspect > target_aspect:
            # Image is wider - crop width, scale height
            scale = target_height / orig_height
            scaled_width = int(orig_width * scale)
            scaled_image = pil_image.resize(
                (scaled_width, target_height),
                Image.Resampling.LANCZOS
            )

            # Center crop to target width
            left = (scaled_width - target_width) // 2
            prepared = scaled_image.crop((
                left,
                0,
                left + target_width,
                target_height
            ))
            logger.info(f"   Cropped width: {target_width}x{target_height}")

        else:
            # Image is taller - crop height, scale width
            scale = target_width / orig_width
            scaled_height = int(orig_height * scale)
            scaled_image = pil_image.resize(
                (target_width, scaled_height),
                Image.Resampling.LANCZOS
            )

            # Center crop to target height
            top = (scaled_height - target_height) // 2
            prepared = scaled_image.crop((
                0,
                top,
                target_width,
                top + target_height
            ))
            logger.info(f"   Cropped height: {target_width}x{target_height}")

        return prepared

    def _generate_with_ai_image(
            self,
            scene_num: int,
            duration: float,
            image_path: Path,
            text_overlay: str,
            output_dir: Path
    ) -> Optional[Path]:
        """Generate video from AI image with Ken Burns effect"""
        try:
            # Load and prepare image to exact Config dimensions
            pil_image = self._load_and_prepare_image(image_path)

            # Verify dimensions
            if pil_image.size != (self.width, self.height):
                logger.error(
                    f"❌ Image preparation failed: got {pil_image.size}, "
                    f"expected ({self.width}, {self.height})"
                )
                return None

            # Use common generation logic
            return self._generate_video_from_image(
                scene_num=scene_num,
                duration=duration,
                pil_image=pil_image,
                text_overlay=text_overlay,
                output_dir=output_dir
            )

        except Exception as e:
            logger.error(f"❌ AI image video generation error: {str(e)}")
            return None

    def _generate_with_theme(
            self,
            scene_num: int,
            duration: float,
            theme: str,
            text_overlay: str,
            output_dir: Path
    ) -> Optional[Path]:
        """Fallback: Generate with colored background"""
        try:
            logger.info(f"   Using fallback theme generation")

            # Theme colors
            themes = {
                'professional blue': (59, 130, 246),
                'green': (34, 197, 94),
                'purple': (168, 85, 247),
                'red': (239, 68, 68),
                'orange': (249, 115, 22),
                'teal': (45, 212, 191)
            }

            color = themes.get(theme.lower(), (59, 130, 246))

            # Create solid color image at exact Config dimensions
            solid_image = Image.new('RGB', (self.width, self.height), color)

            # Use common generation logic
            return self._generate_video_from_image(
                scene_num=scene_num,
                duration=duration,
                pil_image=solid_image,
                text_overlay=text_overlay,
                output_dir=output_dir
            )

        except Exception as e:
            logger.error(f"❌ Theme video generation error: {str(e)}")
            return None

    def _generate_video_from_image(
            self,
            scene_num: int,
            duration: float,
            pil_image: Image.Image,
            text_overlay: str,
            output_dir: Path
    ) -> Optional[Path]:
        """
        Common method: Generate video from PIL Image with Ken Burns effect
        FIXED: Image is already at Config dimensions - no resizing during effect
        """
        try:
            # Verify input dimensions
            if pil_image.size != (self.width, self.height):
                logger.error(
                    f"❌ Input image size mismatch: {pil_image.size} != "
                    f"({self.width}, {self.height})"
                )
                return None

            # Calculate frames using synced duration
            total_frames = int(duration * self.fps)
            logger.info(f"   Generating {total_frames} frames with Ken Burns effect...")

            # Setup video writer with FIXED dimensions
            output_path = output_dir / f"scene_{scene_num}_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                (self.width, self.height)  # FIXED dimensions
            )

            if not writer.isOpened():
                logger.error(f"❌ Failed to open video writer")
                return None

            # Generate frames with Ken Burns effect
            for frame_idx in range(total_frames):
                progress = frame_idx / total_frames if total_frames > 0 else 0

                # Apply Ken Burns effect (already at correct dimensions)
                frame = self._apply_ken_burns(
                    image=pil_image,
                    progress=progress,
                    frame_idx=frame_idx,
                    total_frames=total_frames
                )

                # Verify frame dimensions before adding overlay
                if frame.size != (self.width, self.height):
                    logger.warning(
                        f"⚠️ Frame {frame_idx} size mismatch, resizing: "
                        f"{frame.size} → ({self.width}, {self.height})"
                    )
                    frame = frame.resize(
                        (self.width, self.height),
                        Image.Resampling.LANCZOS
                    )

                # Add text overlay
                if text_overlay:
                    frame = self._add_text_overlay(frame, text_overlay, progress)

                # Convert to cv2 format and write
                frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

                # Final dimension check
                if frame_cv.shape[1] != self.width or frame_cv.shape[0] != self.height:
                    logger.error(
                        f"❌ Frame dimensions mismatch at write: "
                        f"{frame_cv.shape[1]}x{frame_cv.shape[0]}"
                    )
                    writer.release()
                    return None

                writer.write(frame_cv)

                # Progress logging
                if total_frames > 0 and frame_idx % max(1, total_frames // 5) == 0:
                    percent = int(progress * 100)
                    logger.info(f"   Frame {frame_idx}/{total_frames} ({percent}%)")

            writer.release()

            # Verify output
            if not output_path.exists():
                logger.error(f"❌ Video file not created")
                return None

            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Video generated: {output_path.name} ({file_size:.2f} MB)")

            return output_path

        except Exception as e:
            logger.error(f"❌ Video generation error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _apply_ken_burns(
            self,
            image: Image.Image,
            progress: float,
            frame_idx: int,
            total_frames: int
    ) -> Image.Image:
        """
        Apply Ken Burns effect with EASING and AUDIO-AWARE MOTION
        FIXED: Corrected clamping logic to maintain exact output dimensions

        Motion profile:
        - First 15%: Gentle ease-in (establishing shot)
        - Middle 70%: Subtle continuous motion (during narration)
        - Last 15%: Gentle ease-out (prepare for transition)
        """
        if not Config.KEN_BURNS_ENABLED:
            return image

        # Image is already at target dimensions
        assert image.size == (self.width, self.height), \
            f"Ken Burns input size mismatch: {image.size}"

        # Get current preset
        preset = self._current_motion_preset or {}
        zoom_factor = preset.get('zoom_factor', Config.KEN_BURNS_ZOOM_FACTOR)
        pan_intensity = preset.get('pan_intensity', 1.0)
        easing_name = preset.get('easing', Config.KEN_BURNS_EASING)

        # Import and apply easing function
        from src.easing_functions import get_easing_function
        easing_func = get_easing_function(easing_name)
        eased_progress = easing_func(progress)

        # Audio-aware motion modulation
        if Config.KEN_BURNS_AUDIO_AWARE:
            motion_intensity = self._calculate_motion_intensity(progress) * pan_intensity
        else:
            motion_intensity = pan_intensity

        # Calculate zoom with easing (1.0 → zoom_factor)
        zoom_start = 1.0
        current_zoom = zoom_start + (zoom_factor - zoom_start) * eased_progress

        # Calculate zoomed dimensions
        zoomed_w = int(self.width * current_zoom)
        zoomed_h = int(self.height * current_zoom)

        # Resize image to zoomed size
        zoomed_image = image.resize(
            (zoomed_w, zoomed_h),
            Image.Resampling.LANCZOS
        )

        # Calculate pan with motion intensity modulation
        max_pan_x = int(Config.KEN_BURNS_PAN_PIXELS * motion_intensity)
        max_pan_y = int(Config.KEN_BURNS_PAN_PIXELS * 0.5 * motion_intensity)

        # Apply eased pan
        pan_x = int(max_pan_x * eased_progress)
        pan_y = int(max_pan_y * eased_progress)

        # ==================== CORRECTED CLAMPING LOGIC ====================
        # Calculate initial crop box
        left = pan_x
        top = pan_y
        right = left + self.width
        bottom = top + self.height

        # Clamp horizontally while maintaining width
        if right > zoomed_w:
            right = zoomed_w
            left = right - self.width  # Recalculate to maintain width
        if left < 0:
            left = 0
            right = left + self.width  # Recalculate to maintain width

        # Clamp vertically while maintaining height
        if bottom > zoomed_h:
            bottom = zoomed_h
            top = bottom - self.height  # Recalculate to maintain height
        if top < 0:
            top = 0
            bottom = top + self.height  # Recalculate to maintain height

        # Additional safety check for edge cases
        left = max(0, min(left, zoomed_w - self.width))
        top = max(0, min(top, zoomed_h - self.height))
        right = left + self.width
        bottom = top + self.height
        # ==================================================================

        # Crop and return - guaranteed to be target dimensions
        frame = zoomed_image.crop((left, top, right, bottom))

        # Verify output dimensions
        assert frame.size == (self.width, self.height), \
            f"Ken Burns output size mismatch: {frame.size}"

        return frame

    def _calculate_motion_intensity(self, progress: float) -> float:
        """
        Calculate motion intensity based on audio timing

        Motion profile for natural narration:
        - 0-15%: Gentle start (0.3 → 0.6 intensity)
        - 15-85%: Full motion (0.6 → 1.0 intensity)
        - 85-100%: Gentle end (1.0 → 0.4 intensity)
        """
        if progress < 0.15:
            # Ease in: gentle start
            local_progress = progress / 0.15
            return 0.3 + (0.3 * local_progress)  # 0.3 → 0.6

        elif progress < 0.85:
            # Main motion: subtle continuous movement
            local_progress = (progress - 0.15) / 0.7
            return 0.6 + (0.4 * local_progress)  # 0.6 → 1.0

        else:
            # Ease out: gentle end
            local_progress = (progress - 0.85) / 0.15
            return 1.0 - (0.6 * local_progress)  # 1.0 → 0.4

    def _add_text_overlay(
            self,
            frame: Image.Image,
            text: str,
            progress: float
    ) -> Image.Image:
        """Add text overlay with fade in/out"""
        # Calculate opacity based on progress
        if progress < 0.15:
            # Fade in
            opacity = int(255 * (progress / 0.15))
        elif progress > 0.85:
            # Fade out
            opacity = int(255 * ((1 - progress) / 0.15))
        else:
            # Full opacity
            opacity = 255

        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", Config.DEFAULT_FONT_SIZE)
        except:
            try:
                # Try common font paths
                import platform
                if platform.system() == "Windows":
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", Config.DEFAULT_FONT_SIZE)
                else:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                              Config.DEFAULT_FONT_SIZE)
            except:
                font = ImageFont.load_default()
                logger.warning("⚠️ Using fallback font")

        # Create overlay for text
        overlay = Image.new('RGBA', frame.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Get text size
        bbox = overlay_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center text horizontally, bottom 15% vertically
        x = (self.width - text_width) // 2
        y = int(self.height * 0.85)

        # Draw semi-transparent background box for readability
        padding = 20
        box_coords = [
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        ]

        # Draw rounded box
        overlay_draw.rounded_rectangle(
            box_coords,
            radius=10,
            fill=(0, 0, 0, int(180 * (opacity / 255)))  # Semi-transparent black
        )

        # Draw text with stroke for better readability
        stroke_width = 2
        for adj_x in range(-stroke_width, stroke_width + 1):
            for adj_y in range(-stroke_width, stroke_width + 1):
                overlay_draw.text(
                    (x + adj_x, y + adj_y),
                    text,
                    fill=(0, 0, 0, opacity),  # Black outline
                    font=font
                )

        # Main text
        overlay_draw.text(
            (x, y),
            text,
            fill=(255, 255, 255, opacity),  # White text
            font=font
        )

        # Composite overlay onto frame
        frame_rgba = frame.convert('RGBA')
        frame_rgba.alpha_composite(overlay)

        return frame_rgba.convert('RGB')
 