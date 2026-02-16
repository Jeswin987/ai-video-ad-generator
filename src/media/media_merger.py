"""
Stage 5: Merge Video + Audio using FFmpeg
Fixed method naming consistency and enhanced error handling
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional
from src.core.config import Config

logger = logging.getLogger(__name__)


class MediaMerger:
    """Merge video and audio with comprehensive error handling"""

    def __init__(self):
        self._validate_ffmpeg()
        logger.info("✅ MediaMerger ready (FFmpeg)")

    def _validate_ffmpeg(self):
        """Validate FFmpeg is installed and accessible"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True,
                text=True
            )
            version_line = result.stdout.split('\n')[0]
            logger.info(f"📦 {version_line}")
        except FileNotFoundError:
            raise RuntimeError("❌ FFmpeg not installed! Please install FFmpeg first.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"❌ FFmpeg validation failed: {str(e)}")

    def merge_scene(
            self,
            video_path: Path,
            audio_path: Path,
            output_filename: str
    ) -> Optional[Path]:
        """
        Merge video and audio for a single scene
        """
        if not video_path.exists():
            logger.error(f"❌ Video file not found: {video_path}")
            return None

        if not audio_path.exists():
            logger.error(f"❌ Audio file not found: {audio_path}")
            return None

        output_path = Config.OUTPUT_DIR / output_filename

        logger.info(f"🔗 Merging {video_path.name} + {audio_path.name}...")

        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',  # Copy video codec (no re-encoding)
                '-c:a', 'aac',  # Encode audio as AAC
                '-b:a', '128k',  # Audio bitrate
                '-shortest',  # Match shortest stream
                '-loglevel', Config.FFMPEG_LOGLEVEL,
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True
            )

            # Verify output file was created
            if not output_path.exists():
                logger.error(f"❌ Output file not created: {output_path}")
                return None

            # Check file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Merged: {output_path.name} ({file_size_mb:.2f} MB)")

            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"❌ FFmpeg merge error: {error_msg}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during merge: {str(e)}")
            return None

    def concatenate_videos(
            self,
            video_paths: List[Path],
            output_filename: str
    ) -> Optional[Path]:
        """
        Concatenate multiple videos into one final video
        Fixed method name (was 'concatenate' in original)
        """
        if not video_paths:
            logger.error("❌ No videos to concatenate")
            return None

        # Validate all input files exist
        missing_files = [v for v in video_paths if not v.exists()]
        if missing_files:
            logger.error(f"❌ Missing video files: {missing_files}")
            return None

        logger.info(f"🎬 Concatenating {len(video_paths)} videos...")

        try:
            output_path = Config.OUTPUT_DIR / output_filename

            # Create concat file list
            concat_file = Config.TEMP_DIR / "concat_list.txt"
            with open(concat_file, 'w', encoding='utf-8') as f:
                for video_path in video_paths:
                    # Use absolute path and escape special characters
                    abs_path = video_path.resolve()
                    f.write(f"file '{abs_path}'\n")

            logger.info(f"📄 Concat list: {concat_file}")

            # Concatenate using ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', Config.VIDEO_CODEC,
                '-preset', Config.VIDEO_PRESET,
                '-b:v', Config.VIDEO_BITRATE,
                '-c:a', 'aac',
                '-b:a', '128k',
                '-loglevel', Config.FFMPEG_LOGLEVEL,
                str(output_path)
            ]

            logger.info("⚙️ Running FFmpeg concatenation...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True
            )

            # Verify output
            if not output_path.exists():
                logger.error(f"❌ Output file not created: {output_path}")
                # Try to read stderr for debugging
                if result.stderr:
                    logger.error(f"FFmpeg stderr: {result.stderr[:500]}")
                return None

            # Get file info
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            # Get duration if possible
            duration = self._get_video_duration(output_path)
            if duration:
                logger.info(f"✅ Final video: {output_path.name} ({file_size_mb:.2f} MB, {duration:.1f}s)")
            else:
                logger.info(f"✅ Final video: {output_path.name} ({file_size_mb:.2f} MB)")

            # Cleanup concat file
            try:
                concat_file.unlink()
            except Exception as e:
                logger.warning(f"⚠️ Could not delete concat file: {str(e)}")

            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"❌ FFmpeg concatenation error: {error_msg}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during concatenation: {str(e)}")
            return None

    def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """
        Get video duration using ffprobe
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                text=True
            )

            duration = float(result.stdout.strip())
            return duration

        except Exception as e:
            logger.warning(f"⚠️ Could not get video duration: {str(e)}")
            return None

    def add_fade_transitions(
            self,
            video_path: Path,
            output_filename: str,
            fade_duration: float = 0.5
    ) -> Optional[Path]:
        """
        Add fade in/out transitions to video
        """
        if not video_path.exists():
            logger.error(f"❌ Video file not found: {video_path}")
            return None

        duration = self._get_video_duration(video_path)
        if duration is None:
            logger.warning("⚠️ Cannot add fades without duration info")
            return video_path

        logger.info(f"✨ Adding fade transitions ({fade_duration}s)...")

        try:
            output_path = Config.OUTPUT_DIR / output_filename

            # Calculate fade out start time
            fade_out_start = duration - fade_duration

            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', f'fade=t=in:st=0:d={fade_duration},fade=t=out:st={fade_out_start}:d={fade_duration}',
                '-af', f'afade=t=in:st=0:d={fade_duration},afade=t=out:st={fade_out_start}:d={fade_duration}',
                '-c:v', Config.VIDEO_CODEC,
                '-preset', Config.VIDEO_PRESET,
                '-c:a', 'aac',
                '-loglevel', Config.FFMPEG_LOGLEVEL,
                str(output_path)
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            logger.info(f"✅ Fades added: {output_path.name}")
            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"❌ Fade transition error: {error_msg}")
            return video_path
        except Exception as e:
            logger.error(f"❌ Unexpected error adding fades: {str(e)}")
            return video_path

    def extract_audio(
            self,
            video_path: Path,
            output_filename: str
    ) -> Optional[Path]:
        """
        Extract audio from video file
        """
        if not video_path.exists():
            logger.error(f"❌ Video file not found: {video_path}")
            return None

        logger.info(f"🎵 Extracting audio from {video_path.name}...")

        try:
            output_path = Config.OUTPUT_DIR / output_filename

            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'aac',
                '-b:a', '192k',
                '-loglevel', Config.FFMPEG_LOGLEVEL,
                str(output_path)
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            logger.info(f"✅ Audio extracted: {output_path.name}")
            return output_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"❌ Audio extraction error: {error_msg}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error extracting audio: {str(e)}")
            return None
