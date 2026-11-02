"""Tests for src/core/measurement.py, with no GPU: CUDA is faked or reported unavailable."""

import logging
import re

import pytest
import torch

from src.core import measurement
from src.core.measurement import GB, JobMeasurement, Stage, setup_logging


@pytest.fixture(autouse=True)
def clean_measurement_state(monkeypatch):
    monkeypatch.setattr(measurement, "_active_stages", [])
    monkeypatch.setattr(measurement, "_current_job", None)


@pytest.fixture
def no_cuda(monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("CUDA memory stats used without CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", must_not_be_called)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", must_not_be_called)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", must_not_be_called)


class FakeCuda:
    """Fake peak counters: reset sets them to the current values."""

    def __init__(self, monkeypatch):
        self.allocated = 0
        self.reserved = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.resets = 0
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", self.reset)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: self.peak_allocated)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: self.peak_reserved)

    def reset(self):
        self.resets += 1
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def use(self, allocated_gb, reserved_gb):
        self.allocated = int(allocated_gb * GB)
        self.reserved = int(reserved_gb * GB)
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)


@pytest.fixture
def fake_cuda(monkeypatch):
    return FakeCuda(monkeypatch)


def messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "src.core.measurement"]


def test_stage_without_cuda_logs_time_only(no_cuda, caplog):
    caplog.set_level(logging.INFO, logger="src.core.measurement")

    with Stage("Audio Merge"):
        pass

    logged = messages(caplog)
    assert len(logged) == 1
    assert logged[0].startswith("[Timing] Audio Merge: ")
    assert logged[0].endswith(" seconds")


def test_stage_with_fake_cuda_logs_peak_gb(fake_cuda, caplog):
    caplog.set_level(logging.INFO, logger="src.core.measurement")

    stage = Stage("Image Generation").start()
    fake_cuda.use(allocated_gb=1.5, reserved_gb=2.25)
    stage.stop()

    assert fake_cuda.resets == 1
    assert (stage.peak_allocated_gb, stage.peak_reserved_gb) == (1.5, 2.25)
    assert "[VRAM] Image Generation: peak allocated 1.50 GB, peak reserved 2.25 GB" in messages(caplog)


def test_nested_stage_keeps_outer_peak(fake_cuda):
    outer = Stage("Image Generation").start()
    fake_cuda.use(allocated_gb=3.0, reserved_gb=3.5)
    fake_cuda.use(allocated_gb=0.5, reserved_gb=1.0)

    # The inner stage resets the global peak; the outer must still see 3.0 / 3.5
    with Stage("SDXL Loading") as inner:
        fake_cuda.use(allocated_gb=2.0, reserved_gb=2.5)
    outer.stop()

    assert (inner.peak_allocated_gb, inner.peak_reserved_gb) == (2.0, 2.5)
    assert (outer.peak_allocated_gb, outer.peak_reserved_gb) == (3.0, 3.5)


def test_job_peak_is_max_over_stages(fake_cuda, caplog, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "Fake GPU")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index=0: type("Props", (), {"total_memory": 4 * GB})()
    )
    caplog.set_level(logging.INFO, logger="src.core.measurement")

    job = JobMeasurement("job-1").start()
    with Stage("Gemma Loading"):
        fake_cuda.use(allocated_gb=2.0, reserved_gb=3.0)
    fake_cuda.use(allocated_gb=0.0, reserved_gb=0.0)
    with Stage("Image Generation"):
        fake_cuda.use(allocated_gb=2.5, reserved_gb=2.75)
    job.finish()

    logged = messages(caplog)
    assert "[Job] GPU: Fake GPU, total VRAM 4.00 GB" in logged
    assert any(m.startswith("[Timing] JOB TOTAL (incl. Gemma loading): ") for m in logged)
    assert "[VRAM] JOB PEAK (max over stages): allocated 2.50 GB, reserved 3.00 GB" in logged


def test_job_without_cuda_logs_header_and_time_only(no_cuda, caplog):
    caplog.set_level(logging.INFO, logger="src.core.measurement")

    JobMeasurement("job-2").start().finish()

    logged = messages(caplog)
    assert "[Job] GPU: none (CUDA not available)" in logged
    config_lines = [m for m in logged if m.startswith("[Job] Config: scenes=")]
    assert len(config_lines) == 1
    # image_generator.py hard-codes guidance_scale, so the header must not claim a config value
    assert "guidance" not in config_lines[0]
    assert not any(m.startswith("[VRAM]") for m in logged)


def test_setup_logging_writes_run_log_into_given_folder(tmp_path):
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    log_dir = tmp_path / "logs"

    log_path = setup_logging(log_dir)
    try:
        logging.getLogger("test").info("hello run log")
        for handler in root.handlers:
            handler.flush()

        assert log_path.parent == log_dir
        assert re.fullmatch(r"run_\d{8}_\d{6}\.log", log_path.name)
        assert "hello run log" in log_path.read_text(encoding="utf-8")
        # Console logging is kept: the existing handlers are still attached
        assert all(h in root.handlers for h in handlers_before)
    finally:
        for handler in list(root.handlers):
            if handler not in handlers_before:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(level_before)
