"""
Optimized Configuration for 4GB VRAM (RTX 2050) + 8GB RAM
Enhanced for low-memory systems with aggressive optimization
OPTIMIZED: Using Gemma-2-2B-IT + SDXL-Turbo @ 512x512
"""

import os
import logging
from pathlib import Path
import torch
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class Config:
    """Central configuration management with validation"""

    # ============================================================================
    # PROJECT PATHS
    # ============================================================================
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = BASE_DIR / "models"
    ASSETS_DIR = BASE_DIR / "assets"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"
    LOGS_DIR = BASE_DIR / "logs"

    # ============================================================================
    # TEXT MODEL CONFIGURATION (Gemma-2-2B-IT for 4GB VRAM)
    # ============================================================================
    HF_TRUST_REMOTE_CODE = True
    HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/gemma-2-2b-it")
    HF_MODEL_PATH = MODELS_DIR / "gemma-2-2b-it"

    # Model generation settings - OPTIMIZED FOR 4GB
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "800"))  # Reduced from 1000
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    TOP_K = int(os.getenv("TOP_K", "50"))
    REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

    # Memory-optimized settings - CRITICAL FOR 4GB
    USE_4BIT_QUANTIZATION = True  # Essential for 4GB VRAM
    BATCH_SIZE = 1
    USE_FP16 = True
    GRADIENT_CHECKPOINTING = True

    # Retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))

    # ============================================================================
    # IMAGE GENERATION CONFIGURATION (SDXL-Turbo) - OPTIMIZED FOR 4GB
    # ============================================================================
    USE_IMAGE_GENERATION = os.getenv("USE_IMAGE_GENERATION", "true").lower() == "true"
    IMAGE_MODEL_PATH = MODELS_DIR / "sdxl-turbo"

    # Image dimensions - 512x512 is OPTIMAL for 4GB VRAM
    IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "512"))
    IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "512"))
    IMAGE_INFERENCE_STEPS = int(os.getenv("IMAGE_INFERENCE_STEPS", "6"))  # Reduced from 8
    IMAGE_GUIDANCE_SCALE = float(os.getenv("IMAGE_GUIDANCE_SCALE", "0.0"))  # Turbo uses 0

    # ============================================================================
    # IMAGE UPSCALING CONFIGURATION - DISABLED FOR 4GB
    # ============================================================================
    ENABLE_UPSCALING = False  # DISABLED - would cause OOM on 4GB
    UPSCALE_METHOD = os.getenv("UPSCALE_METHOD", "lanczos")
    UPSCALE_FACTOR = float(os.getenv("UPSCALE_FACTOR", "2.0"))
    UPSCALE_BEFORE_VIDEO = False

    # ============================================================================
    # VIDEO CONFIGURATION - OPTIMIZED FOR 4GB
    # ============================================================================
    # Video dimensions (match image dimensions)
    VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "512"))
    VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "512"))
    VIDEO_FPS = int(os.getenv("VIDEO_FPS", "24"))

    # Video encoding settings - FASTEST PRESET FOR 8GB RAM
    VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "1500k")  # Reduced from 2000k
    VIDEO_CODEC = os.getenv("VIDEO_CODEC", "libx264")
    VIDEO_PRESET = os.getenv("VIDEO_PRESET", "ultrafast")  # FASTEST (was "medium")

    # Scene configuration - REDUCED FOR FASTER GENERATION
    NUM_SCENES = int(os.getenv("NUM_SCENES", "3"))  # Reduced from 5
    VIDEO_DURATION_MIN = int(os.getenv("VIDEO_DURATION_MIN", "15"))  # Reduced from 20
    VIDEO_DURATION_MAX = int(os.getenv("VIDEO_DURATION_MAX", "20"))  # Reduced from 30

    # ============================================================================
    # KEN BURNS EFFECT CONFIGURATION - OPTIMIZED
    # ============================================================================
    KEN_BURNS_ENABLED = os.getenv("KEN_BURNS_ENABLED", "true").lower() == "true"
    KEN_BURNS_ZOOM_FACTOR = float(os.getenv("KEN_BURNS_ZOOM_FACTOR", "1.08"))  # Reduced from 1.1
    KEN_BURNS_PAN_PIXELS = int(os.getenv("KEN_BURNS_PAN_PIXELS", "20"))  # Reduced from 30
    KEN_BURNS_EASING = os.getenv("KEN_BURNS_EASING", "ease_in_out_cubic")
    KEN_BURNS_AUDIO_AWARE = os.getenv("KEN_BURNS_AUDIO_AWARE", "true").lower() == "true"

    # ============================================================================
    # AUDIO CONFIGURATION
    # ============================================================================
    # TTS Settings
    TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en")
    TTS_SLOW = os.getenv("TTS_SLOW", "false").lower() == "true"
    TTS_RETRY_ATTEMPTS = int(os.getenv("TTS_RETRY_ATTEMPTS", "3"))

    # Audio mixing
    AUDIO_CROSSFADE_MS = int(os.getenv("AUDIO_CROSSFADE_MS", "250"))  # Reduced from 300
    AUDIO_NORMALIZE = os.getenv("AUDIO_NORMALIZE", "true").lower() == "true"

    # ============================================================================
    # AUDIO-VIDEO SYNC CONFIGURATION
    # ============================================================================
    DURATION_SYNC_ENABLED = os.getenv("DURATION_SYNC_ENABLED", "true").lower() == "true"
    DURATION_SYNC_TOLERANCE = float(os.getenv("DURATION_SYNC_TOLERANCE", "0.5"))
    ALLOW_DURATION_ADJUSTMENT = os.getenv("ALLOW_DURATION_ADJUSTMENT", "false").lower() == "true"
    MIN_SCENE_DURATION = float(os.getenv("MIN_SCENE_DURATION", "2.0"))
    MAX_SCENE_DURATION = float(os.getenv("MAX_SCENE_DURATION", "12.0"))  # Reduced from 15.0

 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # FFmpeg Settings
    FFMPEG_LOGLEVEL = os.getenv("FFMPEG_LOGLEVEL", "error")

    # Font settings
    DEFAULT_FONT_SIZE = int(os.getenv("DEFAULT_FONT_SIZE", "40"))  # Reduced from 48
    FALLBACK_FONT_ENABLED = True

    # ============================================================================
    # CLASS METHODS
    # ============================================================================

    @classmethod
    def find_model_path(cls) -> Path:
        """Auto-find correct model path"""
        # Strategy 1: Direct model folder
        direct_path = cls.MODELS_DIR / "gemma-2-2b-it"
        if direct_path.exists():
            config_file = direct_path / "config.json"
            if config_file.exists():
                logger.info(f"✅ Found model in direct path: {direct_path}")
                return direct_path

        # Strategy 2: HuggingFace cache structure
        model_names = ["gemma-3-4b-it", "gemma-2-2b-it"]
        for model_name in model_names:
            cache_path = cls.MODELS_DIR / model_name / f"models--google--{model_name}"
            if cache_path.exists():
                snapshots_dir = cache_path / "snapshots"
                if snapshots_dir.exists():
                    snapshot_folders = [f for f in snapshots_dir.iterdir() if f.is_dir()]
                    if snapshot_folders:
                        actual_model_path = snapshot_folders[0]
                        logger.info(f"✅ Found model in snapshot: {actual_model_path}")
                        return actual_model_path

                config_file = cache_path / "config.json"
                if config_file.exists():
                    logger.info(f"✅ Found model in cache: {cache_path}")
                    return cache_path

        logger.warning(f"⚠ Model not found, using configured path: {cls.HF_MODEL_PATH}")
        logger.warning("⚠ Model not found. Please manually download from Hugging Face and place inside the models/ directory.")
        return cls.HF_MODEL_PATH

    @classmethod
    def initialize(cls) -> bool:
        """Initialize all required directories"""
        try:
            directories = [
                cls.MODELS_DIR,
                cls.ASSETS_DIR,
                cls.OUTPUT_DIR,
                cls.TEMP_DIR,
                cls.LOGS_DIR,
                cls.OUTPUT_DIR / "images",
                cls.OUTPUT_DIR / "audio",
                cls.OUTPUT_DIR / "videos",
            ]

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)

                if not os.access(directory, os.W_OK):
                    logger.error(f"❌ No write permission for: {directory}")
                    return False

                logger.info(f"✅ Directory ready: {directory}")

            # Auto-detect model path
            cls.HF_MODEL_PATH = cls.find_model_path()

            # Validate device
            cls._validate_device()

            # Check image generation
            cls._check_image_generation()

            # Log configuration
            cls._log_config_summary()

            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return False

    @classmethod
    def _check_image_generation(cls):
        """Check if SDXL-Turbo is available"""
        if cls.USE_IMAGE_GENERATION:
            if cls.IMAGE_MODEL_PATH.exists():
                logger.info(f"✅ Image generation enabled: {cls.IMAGE_MODEL_PATH}")
            else:
                logger.warning(f"⚠ SDXL-Turbo not found at: {cls.IMAGE_MODEL_PATH}")
                logger.warning("⚠ SDXL-Turbo not found. Please download it from Hugging Face and place inside the models/ directory.")
                logger.warning(f"⚠ Falling back to colored backgrounds")
                cls.USE_IMAGE_GENERATION = False

    @classmethod
    def _validate_device(cls):
        """Validate GPU/CPU configuration"""
        if cls.DEVICE == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                logger.info(f"✅ GPU: {gpu_name}")
                logger.info(f"✅ VRAM: {vram_total:.2f} GB")

                if vram_total < 4:
                    logger.error("❌ Less than 4GB VRAM - GPU mode not recommended")
                elif vram_total <= 4.5:
                    logger.warning("⚠ 4GB VRAM - using aggressive optimization")
                    logger.info("💡 Recommended: 3 scenes @ 512x512, no upscaling")
                elif vram_total <= 6:
                    logger.info("💡 4-6GB VRAM - current settings are optimal")
                else:
                    logger.info("💡 6GB+ VRAM - can increase to 5 scenes or 768x768")

            except Exception as e:
                logger.warning(f"⚠ Could not get GPU info: {str(e)}")
        else:
            logger.info("ℹ Using CPU (GPU not available)")

    @classmethod
    def _log_config_summary(cls):
        """Log configuration summary"""
        logger.info("=" * 70)
        logger.info("CONFIGURATION SUMMARY - OPTIMIZED FOR 4GB VRAM")
        logger.info("=" * 70)
        logger.info(f"Text Model: {cls.HF_MODEL_NAME}")
        logger.info(f"Text Model Path: {cls.HF_MODEL_PATH}")
        logger.info(f"Device: {cls.DEVICE}")
        logger.info(f"Max Tokens: {cls.MAX_NEW_TOKENS}")
        logger.info("-" * 70)
        logger.info(f"Image Generation: {'Enabled' if cls.USE_IMAGE_GENERATION else 'Disabled'}")
        if cls.USE_IMAGE_GENERATION:
            logger.info(f"Image Model Path: {cls.IMAGE_MODEL_PATH}")
            logger.info(f"Image Size: {cls.IMAGE_WIDTH}x{cls.IMAGE_HEIGHT}")
            logger.info(f"Inference Steps: {cls.IMAGE_INFERENCE_STEPS}")
            logger.info(f"Upscaling: {'Enabled' if cls.ENABLE_UPSCALING else 'Disabled (4GB mode)'}")
        logger.info("-" * 70)
        logger.info(f"Video: {cls.VIDEO_WIDTH}x{cls.VIDEO_HEIGHT} @ {cls.VIDEO_FPS}fps")
        logger.info(f"Video Preset: {cls.VIDEO_PRESET} (optimized for 8GB RAM)")
        logger.info(f"Ken Burns: {'Enabled' if cls.KEN_BURNS_ENABLED else 'Disabled'}")
        if cls.KEN_BURNS_ENABLED:
            logger.info(f"Ken Burns Zoom: {cls.KEN_BURNS_ZOOM_FACTOR}x")
            logger.info(f"Audio-Aware Motion: {'Yes' if cls.KEN_BURNS_AUDIO_AWARE else 'No'}")
        logger.info(f"Scenes: {cls.NUM_SCENES} (optimized for 4GB)")
        logger.info(f"Duration: {cls.VIDEO_DURATION_MIN}-{cls.VIDEO_DURATION_MAX}s")
        logger.info("-" * 70)
        logger.info(f"Audio Crossfade: {cls.AUDIO_CROSSFADE_MS}ms")
        logger.info(f"Duration Sync: {'Enabled' if cls.DURATION_SYNC_ENABLED else 'Disabled'}")
        logger.info(f"Scene Duration Range: {cls.MIN_SCENE_DURATION}s - {cls.MAX_SCENE_DURATION}s")
        logger.info("=" * 70)
        logger.info("💡 Estimated generation time: 3-5 minutes per video")
        logger.info("=" * 70)

    @classmethod
    def cleanup_temp_files(cls) -> int:
        """Clean up temporary files"""
        try:
            count = 0
            for temp_file in cls.TEMP_DIR.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    count += 1

            if count > 0:
                logger.info(f"🧹 Cleaned up {count} temporary files")

            return count

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {str(e)}")
            return 0

    @classmethod
    def get_disk_space_gb(cls, path: Optional[Path] = None) -> float:
        """Get available disk space in GB"""
        try:
            if path is None:
                path = cls.BASE_DIR

            import shutil
            stat = shutil.disk_usage(path)
            available_gb = stat.free / (1024 ** 3)
            return available_gb

        except Exception as e:
            logger.warning(f"⚠ Could not check disk space: {str(e)}")
            return 0.0

    @classmethod
    def validate_disk_space(cls, required_gb: float = 5.0) -> bool:
        """Validate sufficient disk space"""
        available = cls.get_disk_space_gb()

        if available < required_gb:
            logger.error(f"❌ Insufficient disk space: {available:.2f}GB available, {required_gb}GB required")
            return False

        logger.info(f"✅ Disk space OK: {available:.2f}GB available")
        return True



if __name__ == "__main__":
    try:
        Config.initialize()
    except Exception as e:
        logger.warning(f"⚠ Auto-initialization skipped: {str(e)}")