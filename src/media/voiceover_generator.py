"""
Stage 3: Generate voiceovers from scene scripts
Enhanced with audio duration calculation, quality settings, and retry logic
WITH DURATION SYNC SUPPORT
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from gtts import gTTS
from src.core.config import Config

logger = logging.getLogger(__name__)


class VoiceoverGenerator:
    """Generate voiceovers with duration management and retry logic"""

    def __init__(self):
        self._validate_dependencies()
        logger.info("✅ VoiceoverGenerator ready (gTTS)")

    def _validate_dependencies(self):
        """Check if ffprobe is available for audio duration calculation"""
        try:
            subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
            self.has_ffprobe = True
        except:
            logger.warning("⚠️ ffprobe not found - duration calculation will be estimated")
            self.has_ffprobe = False

    def get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """
        Get audio duration in seconds using ffprobe
        Falls back to estimation if ffprobe not available
        """
        if not audio_path.exists():
            logger.error(f"❌ Audio file not found: {audio_path}")
            return None

        if self.has_ffprobe:
            try:
                cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(audio_path)
                ]

                result = subprocess.run(cmd, capture_output=True, check=True, text=True)
                duration = float(result.stdout.strip())
                return duration

            except Exception as e:
                logger.warning(f"⚠️ ffprobe failed: {str(e)}, using estimation")

        # Fallback: estimate based on file size (very rough)
        # gTTS typically produces ~2.5 KB per second of audio
        try:
            file_size_kb = audio_path.stat().st_size / 1024
            estimated_duration = file_size_kb / 2.5
            logger.info(f"ℹ️ Estimated duration: {estimated_duration:.2f}s")
            return estimated_duration
        except Exception as e:
            logger.error(f"❌ Could not estimate duration: {str(e)}")
            return None

    def generate(self, scene: Dict[str, Any], output_filename: str) -> Optional[Path]:
        """
        Generate voiceover with retry logic
        """
        script = scene.get('voiceover_script', '')
        scene_num = scene.get('scene_number', 1)

        if not script or not script.strip():
            logger.warning(f"⚠️ No script for Scene {scene_num}")
            return None

        logger.info(f"🎤 Generating voiceover for Scene {scene_num}...")
        logger.info(f"📝 Script: {script[:80]}...")

        audio_path = Config.OUTPUT_DIR / output_filename

        # Retry loop for gTTS (network-dependent)
        for attempt in range(1, Config.TTS_RETRY_ATTEMPTS + 1):
            try:
                logger.info(f"🔄 TTS attempt {attempt}/{Config.TTS_RETRY_ATTEMPTS}")

                # Generate speech
                tts = gTTS(
                    text=script,
                    lang=Config.TTS_LANGUAGE,
                    slow=Config.TTS_SLOW
                )

                # Save to file
                tts.save(str(audio_path))

                # Verify file was created
                if not audio_path.exists():
                    logger.error(f"❌ Audio file not created: {audio_path}")
                    if attempt < Config.TTS_RETRY_ATTEMPTS:
                        time.sleep(Config.RETRY_DELAY)
                        continue
                    return None

                # Get duration
                duration = self.get_audio_duration(audio_path)
                if duration:
                    logger.info(f"✅ Voiceover saved: {audio_path} ({duration:.2f}s)")
                else:
                    logger.info(f"✅ Voiceover saved: {audio_path}")

                return audio_path

            except Exception as e:
                logger.error(f"❌ TTS error on attempt {attempt}: {str(e)}")
                if attempt < Config.TTS_RETRY_ATTEMPTS:
                    logger.info(f"⏳ Retrying in {Config.RETRY_DELAY}s...")
                    time.sleep(Config.RETRY_DELAY)
                    continue

        logger.error(f"❌ Failed after {Config.TTS_RETRY_ATTEMPTS} attempts")
        return None

    def generate_with_duration_sync(
            self,
            scene: Dict[str, Any],
            output_filename: str,
            target_duration: Optional[float] = None,
            tolerance: float = 0.5
    ) -> Tuple[Optional[Path], Optional[float]]:
        """
        Generate voiceover and return both path and measured duration
        Optionally adjusts to target duration if specified

        Args:
            scene: Scene dictionary with voiceover_script
            output_filename: Output MP3 filename
            target_duration: Optional target duration in seconds
            tolerance: Acceptable duration difference (seconds)

        Returns:
            Tuple of (audio_path, measured_duration)
        """
        # Generate TTS
        audio_path = self.generate(scene, output_filename)

        if not audio_path:
            return None, None

        # Measure duration
        measured_duration = self.get_audio_duration(audio_path)

        if not measured_duration:
            logger.warning("⚠️ Could not measure audio duration")
            return audio_path, None

        # If target specified and outside tolerance, adjust
        if target_duration and abs(measured_duration - target_duration) > tolerance:
            logger.info(
                f"🔧 Duration mismatch: {measured_duration:.2f}s vs target {target_duration:.2f}s"
            )
            logger.info(f"🔧 Adjusting audio duration...")

            adjusted_path = self.adjust_duration(audio_path, target_duration)

            if adjusted_path and adjusted_path != audio_path:
                # Re-measure adjusted audio
                final_duration = self.get_audio_duration(adjusted_path)
                if final_duration:
                    logger.info(f"✅ Audio adjusted to {final_duration:.2f}s")
                    return adjusted_path, final_duration

            # If adjustment failed, return original
            logger.warning("⚠️ Using original audio duration")
            return audio_path, measured_duration

        return audio_path, measured_duration

    def adjust_duration(self, audio_path: Path, target_duration: float) -> Optional[Path]:
        """
        Adjust audio duration to match target using ffmpeg
        Returns path to adjusted audio file
        """
        if not audio_path.exists():
            logger.error(f"❌ Audio file not found: {audio_path}")
            return None

        if not self.has_ffprobe:
            logger.warning("⚠️ ffprobe not available, cannot adjust duration accurately")
            return audio_path

        # Get current duration
        current_duration = self.get_audio_duration(audio_path)
        if current_duration is None:
            logger.error("❌ Cannot determine audio duration")
            return audio_path

        # Check if adjustment needed
        tolerance = 0.5  # seconds
        if abs(current_duration - target_duration) < tolerance:
            logger.info(f"✅ Duration OK: {current_duration:.2f}s (target: {target_duration:.2f}s)")
            return audio_path

        logger.info(f"🔧 Adjusting duration from {current_duration:.2f}s to {target_duration:.2f}s")

        try:
            # Calculate speed factor
            speed_factor = current_duration / target_duration

            # Limit speed adjustment to reasonable range
            if speed_factor < 0.5 or speed_factor > 2.0:
                logger.warning(f"⚠️ Speed factor {speed_factor:.2f} out of range, using original")
                return audio_path

            # Create output path
            output_path = audio_path.parent / f"{audio_path.stem}_adjusted{audio_path.suffix}"

            # Use ffmpeg atempo filter for speed adjustment
            # atempo only accepts values between 0.5 and 2.0
            cmd = [
                'ffmpeg', '-y',
                '-i', str(audio_path),
                '-filter:a', f'atempo={speed_factor}',
                '-vn',
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True
            )

            # Verify adjusted file
            adjusted_duration = self.get_audio_duration(output_path)
            if adjusted_duration:
                logger.info(f"✅ Duration adjusted: {adjusted_duration:.2f}s")
                return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return audio_path
        except Exception as e:
            logger.error(f"❌ Adjustment error: {str(e)}")
            return audio_path

    def generate_silence(self, duration: float, output_filename: str) -> Optional[Path]:
        """
        Generate a silent audio file of specified duration
        Useful for scenes without voiceover
        """
        if not self.has_ffprobe:
            logger.warning("⚠️ ffmpeg not available, cannot generate silence")
            return None

        try:
            output_path = Config.OUTPUT_DIR / output_filename
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', f'anullsrc=r=44100:cl=stereo',
                '-t', str(duration),
                '-acodec', 'aac',
                str(output_path)
            ]

            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"✅ Silence generated: {output_path} ({duration:.2f}s)")
            return output_path

        except Exception as e:
            logger.error(f"❌ Failed to generate silence: {str(e)}")
            return None

    def concatenate_audio(self, audio_files: list, output_filename: str) -> Optional[Path]:
        """
        Concatenate multiple audio files into one
        """
        if not audio_files:
            logger.error("❌ No audio files to concatenate")
            return None

        if not self.has_ffprobe:
            logger.warning("⚠️ ffmpeg not available, cannot concatenate")
            return None

        try:
            output_path = Config.OUTPUT_DIR / output_filename

            # Create concat file list
            concat_file = Config.TEMP_DIR / "audio_concat.txt"
            with open(concat_file, 'w') as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")

            # Concatenate using ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ]

            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"✅ Audio concatenated: {output_path}")

            # Cleanup
            concat_file.unlink()

            return output_path

        except Exception as e:
            logger.error(f"❌ Concatenation failed: {str(e)}")
            return None
