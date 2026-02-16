"""
Audio Merger - Creates continuous audio across all scenes
Uses FFmpeg directly instead of pydub (Python 3.13 compatible)
"""

import logging
import subprocess
from pathlib import Path
from typing import List
from src.config import Config

logger = logging.getLogger(__name__)

class AudioMerger:
    """Merge multiple audio files using FFmpeg"""
    
    def __init__(self):
        logger.info("🎵 AudioMerger ready (FFmpeg mode)")
    
    def merge_scene_audio(
        self,
        audio_files: List[Path],
        output_path: Path,
        crossfade_ms: int = 300
    ) -> Path:
        """
        Merge multiple audio files with FFmpeg
        
        Args:
            audio_files: List of audio file paths in order
            output_path: Output path for merged audio
            crossfade_ms: Crossfade duration in milliseconds (ignored for now)
        
        Returns:
            Path to merged audio file
        """
        try:
            logger.info(f"🎵 Merging {len(audio_files)} audio files with FFmpeg...")
            
            if not audio_files:
                raise ValueError("No audio files provided")
            
            if len(audio_files) == 1:
                # Just copy single file
                import shutil
                shutil.copy(str(audio_files[0]), str(output_path))
                logger.info(f"✅ Copied single audio: {output_path.name}")
                return output_path
            
            # Create concat file for FFmpeg
            concat_file = output_path.parent / f"concat_{output_path.stem}.txt"
            with open(concat_file, 'w') as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file.absolute()}'\n")
            
            # Use FFmpeg to concatenate
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Clean up concat file
            concat_file.unlink()
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Merged audio: {output_path.name}")
            logger.info(f"   Size: {file_size_mb:.2f}MB")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ FFmpeg merge failed: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"❌ Audio merge failed: {str(e)}")
            raise
    
    def create_silent_audio(
        self,
        duration_ms: int,
        output_path: Path
    ) -> Path:
        """Create silent audio track using FFmpeg"""
        try:
            duration_s = duration_ms / 1000.0
            
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=duration={duration_s}:sample_rate=44100',
                '-c:a', 'libmp3lame',
                '-b:a', '192k',
                '-y',
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"✅ Created silent audio: {duration_s:.1f}s")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Silent audio creation failed: {str(e)}")
            raise
