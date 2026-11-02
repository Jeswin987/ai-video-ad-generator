"""
Measurement helpers: per-run log file, per-stage time and peak VRAM, job header and total.
Measurement only - nothing here changes pipeline behaviour.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

GB = 1024 ** 3
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Stages that have started and not yet stopped. Peak VRAM stats are global to
# the process, so a nested stage folds the peak seen so far into its parents
# before it resets them. One job runs at a time, so one shared list is enough.
_active_stages: List["Stage"] = []
_current_job: Optional["JobMeasurement"] = None


def setup_logging(log_dir: Path) -> Path:
    """Keep the console logging and also write this run to log_dir/run_<YYYYmmdd_HHMMSS>.log"""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    root = logging.getLogger()
    # basicConfig does nothing when handlers already exist, so make sure INFO reaches the file
    if root.getEffectiveLevel() > logging.INFO:
        root.setLevel(logging.INFO)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    logger.info(f"📝 Run log: {log_path}")
    return log_path


def _read_peaks():
    return (torch.cuda.max_memory_allocated() / GB,
            torch.cuda.max_memory_reserved() / GB)


class Stage:
    """
    Logs a stage's elapsed time in the existing [Timing] format and, when CUDA
    is available, its peak VRAM. Use as a context manager or with start()/stop().
    """

    def __init__(self, name: str):
        self.name = name
        self.elapsed = None
        self.peak_allocated_gb = 0.0
        self.peak_reserved_gb = 0.0
        self._start = None
        self._cuda = False

    def _fold_peaks(self, allocated_gb: float, reserved_gb: float):
        self.peak_allocated_gb = max(self.peak_allocated_gb, allocated_gb)
        self.peak_reserved_gb = max(self.peak_reserved_gb, reserved_gb)

    def start(self) -> "Stage":
        self._cuda = torch.cuda.is_available()
        if self._cuda:
            allocated_gb, reserved_gb = _read_peaks()
            for parent in _active_stages:
                parent._fold_peaks(allocated_gb, reserved_gb)
            torch.cuda.reset_peak_memory_stats()
        _active_stages.append(self)
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._start
        if self in _active_stages:
            _active_stages.remove(self)

        logger.info(f"[Timing] {self.name}: {self.elapsed:.2f} seconds")

        if self._cuda:
            allocated_gb, reserved_gb = _read_peaks()
            self._fold_peaks(allocated_gb, reserved_gb)
            for parent in _active_stages:
                parent._fold_peaks(allocated_gb, reserved_gb)
            if _current_job is not None:
                _current_job.record(self)
            logger.info(
                f"[VRAM] {self.name}: peak allocated {self.peak_allocated_gb:.2f} GB, "
                f"peak reserved {self.peak_reserved_gb:.2f} GB"
            )
        return self.elapsed

    def __enter__(self) -> "Stage":
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


class JobMeasurement:
    """Job-wide total time and peak VRAM (the maximum over all stages)."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.peak_allocated_gb = 0.0
        self.peak_reserved_gb = 0.0
        self._start = None

    def record(self, stage: Stage):
        self.peak_allocated_gb = max(self.peak_allocated_gb, stage.peak_allocated_gb)
        self.peak_reserved_gb = max(self.peak_reserved_gb, stage.peak_reserved_gb)

    def start(self) -> "JobMeasurement":
        global _current_job
        # Drop stages a failed earlier job left running
        _active_stages.clear()
        _current_job = self
        self._start = time.perf_counter()
        # Runs before the job's own try block, so a failure here must not stop the job
        try:
            log_job_header(self.job_id)
        except Exception as e:
            logger.warning(f"⚠ Could not log job header: {str(e)}")
        return self

    def finish(self) -> float:
        global _current_job
        total = time.perf_counter() - self._start
        if _current_job is self:
            _current_job = None

        logger.info(f"[Timing] JOB TOTAL (incl. Gemma loading): {total:.2f} seconds")
        if torch.cuda.is_available():
            logger.info(
                f"[VRAM] JOB PEAK (max over stages): allocated {self.peak_allocated_gb:.2f} GB, "
                f"reserved {self.peak_reserved_gb:.2f} GB"
            )
        return total


def log_job_header(job_id: str):
    """Log hardware, library versions and the effective config for one job"""
    from src.core.config import Config

    logger.info("=" * 70)
    logger.info(f"[Job] {job_id}")
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / GB
        logger.info(f"[Job] GPU: {torch.cuda.get_device_name(0)}, total VRAM {total_vram_gb:.2f} GB")
    else:
        logger.info("[Job] GPU: none (CUDA not available)")
    logger.info(f"[Job] torch {torch.__version__}, CUDA build {torch.version.cuda}")
    logger.info(
        f"[Job] Config: scenes={Config.NUM_SCENES}, "
        f"image={Config.IMAGE_WIDTH}x{Config.IMAGE_HEIGHT}, "
        f"steps={Config.IMAGE_INFERENCE_STEPS}, "
        f"video={Config.VIDEO_WIDTH}x{Config.VIDEO_HEIGHT} @ {Config.VIDEO_FPS}fps"
    )
    logger.info("=" * 70)
