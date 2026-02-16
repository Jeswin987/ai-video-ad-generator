"""
Image Upscaler - Optional quality enhancement
Provides multiple upscaling methods with VRAM-aware implementations
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Literal
from PIL import Image

from src.core.config import Config

logger = logging.getLogger(__name__)

UpscaleMethod = Literal["lanczos", "bicubic", "ffmpeg"]


class ImageUpscaler:
    """Handle image upscaling with multiple backend options"""

    def __init__(self):
        self.enabled = Config.ENABLE_UPSCALING
        self.method = Config.UPSCALE_METHOD
        self.factor = Config.UPSCALE_FACTOR

        if self.enabled:
            logger.info(
                f"🔍 Upscaler initialized: {self.method} "
                f"({self.factor}x scaling)"
            )
        else:
            logger.info("🔍 Upscaling disabled")

    def upscale_image(
            self,
            input_path: Path,
            output_path: Optional[Path] = None,
            method: Optional[UpscaleMethod] = None
    ) -> Optional[Path]:
        """
        Upscale image using specified method

        Args:
            input_path: Path to input image
            output_path: Optional output path (defaults to _upscaled suffix)
            method: Upscaling method override

        Returns:
            Path to upscaled image or None on failure
        """
        if not self.enabled:
            logger.debug("Upscaling disabled, returning original")
            return input_path

        if not input_path.exists():
            logger.error(f"❌ Input image not found: {input_path}")
            return None

        # Use default method if not specified
        method = method or self.method

        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}"

        try:
            logger.info(f"🔍 Upscaling {input_path.name} using {method}...")

            if method == "ffmpeg":
                return self._upscale_ffmpeg(input_path, output_path)
            elif method in ["lanczos", "bicubic"]:
                return self._upscale_pil(input_path, output_path, method)
            else:
                logger.warning(f"⚠️ Unknown method: {method}, using lanczos")
                return self._upscale_pil(input_path, output_path, "lanczos")

        except Exception as e:
            logger.error(f"❌ Upscaling failed: {str(e)}")
            return None

    def _upscale_pil(
            self,
            input_path: Path,
            output_path: Path,
            method: str
    ) -> Optional[Path]:
        """
        Upscale using PIL (CPU-based, no VRAM impact)
        Fast and reliable for 2x scaling
        """
        try:
            # Load image
            image = Image.open(input_path)
            orig_width, orig_height = image.size

            # Calculate new dimensions
            new_width = int(orig_width * self.factor)
            new_height = int(orig_height * self.factor)

            logger.info(
                f"   Original: {orig_width}x{orig_height} → "
                f"Upscaled: {new_width}x{new_height}"
            )

            # Select resampling filter
            if method == "lanczos":
                resampling = Image.Resampling.LANCZOS  # Best quality
            elif method == "bicubic":
                resampling = Image.Resampling.BICUBIC  # Good balance
            else:
                resampling = Image.Resampling.LANCZOS

            # Upscale
            upscaled = image.resize(
                (new_width, new_height),
                resample=resampling
            )

            # Save with high quality
            upscaled.save(
                output_path,
                format='PNG',
                optimize=False,
                compress_level=1  # Minimal compression for quality
            )

            # Verify
            if not output_path.exists():
                logger.error("❌ Upscaled image not saved")
                return None

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Upscaled: {output_path.name} ({file_size_mb:.2f}MB)")

            return output_path

        except Exception as e:
            logger.error(f"❌ PIL upscaling failed: {str(e)}")
            return None

    def _upscale_ffmpeg(
            self,
            input_path: Path,
            output_path: Path
    ) -> Optional[Path]:
        """
        Upscale using ffmpeg with high-quality filters
        Slower but produces excellent results
        """
        try:
            # Load to get dimensions
            image = Image.open(input_path)
            orig_width, orig_height = image.size
            new_width = int(orig_width * self.factor)
            new_height = int(orig_height * self.factor)

            logger.info(
                f"   Original: {orig_width}x{orig_height} → "
                f"Upscaled: {new_width}x{new_height}"
            )

            # Build ffmpeg command with high-quality scaling
            # Using Lanczos filter with 3-lobe precision
            cmd = [
                'ffmpeg', '-y',
                '-i', str(input_path),
                '-vf', f'scale={new_width}:{new_height}:flags=lanczos',
                '-q:v', '1',  # Highest quality
                '-pix_fmt', 'rgb24',
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True
            )

            # Verify output
            if not output_path.exists():
                logger.error("❌ ffmpeg did not create output")
                return None

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Upscaled (ffmpeg): {output_path.name} ({file_size_mb:.2f}MB)")

            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ffmpeg upscaling failed: {e.stderr}")
            return None

        except Exception as e:
            logger.error(f"❌ ffmpeg upscaling error: {str(e)}")
            return None

    def batch_upscale(
            self,
            image_paths: dict,
            output_dir: Path
    ) -> dict:
        """
        Upscale multiple images

        Args:
            image_paths: Dict of {scene_num: image_path}
            output_dir: Output directory for upscaled images

        Returns:
            Dict of {scene_num: upscaled_path}
        """
        if not self.enabled:
            logger.info("Upscaling disabled, returning original paths")
            return image_paths

        logger.info(f"🔍 Batch upscaling {len(image_paths)} images...")

        output_dir.mkdir(parents=True, exist_ok=True)
        upscaled_images = {}

        for scene_num, image_path in image_paths.items():
            if not image_path or not image_path.exists():
                logger.warning(f"⚠️ Scene {scene_num}: Image not found, skipping")
                continue

            # Generate output path
            output_path = output_dir / f"scene_{scene_num}_upscaled.png"

            # Upscale
            upscaled_path = self.upscale_image(
                input_path=image_path,
                output_path=output_path
            )

            if upscaled_path:
                upscaled_images[scene_num] = upscaled_path
            else:
                logger.warning(f"⚠️ Scene {scene_num}: Upscaling failed, using original")
                upscaled_images[scene_num] = image_path

        success_rate = len([p for p in upscaled_images.values() if p]) / len(image_paths) * 100
        logger.info(
            f"✅ Batch upscaling complete: "
            f"{len(upscaled_images)}/{len(image_paths)} ({success_rate:.0f}%)"
        )

        return upscaled_images

    def get_upscaled_dimensions(self, original_width: int, original_height: int) -> tuple:
        """Get dimensions after upscaling"""
        if not self.enabled:
            return (original_width, original_height)

        new_width = int(original_width * self.factor)
        new_height = int(original_height * self.factor)
        return (new_width, new_height)
