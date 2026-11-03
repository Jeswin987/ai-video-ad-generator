"""
Each job keeps its own files under output/<job_id>/ (bug B2).

FFmpeg/ffprobe, gTTS, the text models and the video renderer are faked;
OUTPUT_DIR and TEMP_DIR point into tmp_path.
"""

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import app as app_module
import src.core.video_generation_pipeline as pipeline_module
import src.media.media_merger as media_merger_module
import src.media.voiceover_generator as voiceover_module
from src.core.video_generation_pipeline import VideoGenerationPipeline

# Modules whose Config is patched. Another test reloads config.py, so these
# modules may hold a different Config class than src.core.config does now.
CODE_MODULES = [app_module, pipeline_module, media_merger_module, voiceover_module]


class FakeStorylineGenerator:
    def generate(self, prompt):
        return {"product_name": "Widget", "prompt": prompt}


class FakeSceneDescriptor:
    def generate_scenes(self, storyline):
        return [
            {
                "scene_number": n,
                "visual_description": f"Widget on a desk, shot {n}",
                "voiceover_script": f"{storyline['prompt']} scene {n}",
                "text_overlay": f"Widget {n}",
                "duration": 3.0,
            }
            for n in (1, 2)
        ]


class FakeVideoGenerator:
    def generate_scene_video(self, scene, output_dir, visual_theme, scene_image_path):
        path = output_dir / f"scene_{scene['scene_number']}_video.mp4"
        path.write_text(f"video {path}")
        return path


class FakeTTS:
    def __init__(self, text, lang, slow):
        self.text = text

    def save(self, path):
        Path(path).write_text(self.text)


def fail_on_sleep(seconds):
    raise AssertionError(f"time.sleep({seconds}) called during a job")


@pytest.fixture
def env(monkeypatch, tmp_path):
    output_root = tmp_path / "output"
    temp_root = tmp_path / "temp"
    output_root.mkdir()
    temp_root.mkdir()

    settings = {
        "OUTPUT_DIR": output_root,
        "TEMP_DIR": temp_root,
        "USE_IMAGE_GENERATION": False,
        "ENABLE_UPSCALING": False,
        "DURATION_SYNC_ENABLED": True,
        "ALLOW_DURATION_ADJUSTMENT": False,
    }
    for config in {module.Config for module in CODE_MODULES}:
        for name, value in settings.items():
            monkeypatch.setattr(config, name, value)

    state = SimpleNamespace(output_root=output_root, concat_lists=[], ffmpeg_calls=0)

    def fake_run(cmd, **kwargs):
        if "-version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, f"{cmd[0]} version fake", "")
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, "3.0", "")
        if "concat" in cmd:
            state.concat_lists.append(Path(cmd[cmd.index("-i") + 1]))
        # Unique content per call, so an overwrite always changes the file
        state.ffmpeg_calls += 1
        Path(cmd[-1]).write_text(f"ffmpeg output {cmd[-1]} #{state.ffmpeg_calls}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(voiceover_module, "gTTS", FakeTTS)
    monkeypatch.setattr(pipeline_module, "StorylineGenerator", FakeStorylineGenerator)
    monkeypatch.setattr(pipeline_module, "SceneDescriptor", FakeSceneDescriptor)
    monkeypatch.setattr(pipeline_module, "VideoGeneratorWithImages", FakeVideoGenerator)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Nothing on the faked job path should sleep (the old progress loop did)
    monkeypatch.setattr(time, "sleep", fail_on_sleep)
    monkeypatch.setattr(app_module.socketio, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, "jobs", {})
    return state


def run_job(env, job_id):
    output_dir = env.output_root / job_id
    output_dir.mkdir()
    return VideoGenerationPipeline().generate_video(
        prompt=f"Ad for {job_id}", output_dir=output_dir
    )


def snapshot(root):
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_job_writes_every_file_into_its_own_folder(env, tmp_path):
    job_dir = env.output_root / "job-a"

    final_video = run_job(env, "job-a")

    assert final_video == job_dir / "final_video_with_audio.mp4"
    assert (job_dir / "audio" / "audio_scene_1.mp3").exists()
    assert (job_dir / "audio" / "audio_scene_2.mp3").exists()
    written = snapshot(tmp_path)
    assert written
    outside = [p for p in written if job_dir not in p.parents]
    assert outside == []


def test_concat_lists_are_job_specific(env):
    run_job(env, "job-a")
    lists_a = list(env.concat_lists)
    run_job(env, "job-b")
    lists_b = env.concat_lists[len(lists_a):]

    assert lists_a and lists_b
    assert all(env.output_root / "job-a" in p.parents for p in lists_a)
    assert all(env.output_root / "job-b" in p.parents for p in lists_b)


def test_second_job_does_not_overwrite_the_first(env, tmp_path):
    run_job(env, "job-a")
    files_a = snapshot(tmp_path)

    run_job(env, "job-b")
    files_after = snapshot(tmp_path)

    for path, content in files_a.items():
        assert files_after.get(path) == content, f"{path} was changed by job-b"
    new_files = [p for p in files_after if p not in files_a]
    assert new_files
    assert all(env.output_root / "job-b" in p.parents for p in new_files)


def assert_seven_stages_in_order(updates):
    """updates: (progress, message) pairs reported for the pipeline stages."""
    assert [m.split(":")[0] for _, m in updates] == [f"Stage {n}/7" for n in range(1, 8)]
    progress = [p for p, _ in updates]
    assert progress == sorted(set(progress))  # strictly increasing
    assert all(0 < p < 100 for p in progress)


def test_pipeline_reports_all_seven_stages_in_order(env):
    output_dir = env.output_root / "job-a"
    output_dir.mkdir()
    updates = []

    final_video = VideoGenerationPipeline().generate_video(
        prompt="Ad for job-a",
        output_dir=output_dir,
        on_stage=lambda progress, message: updates.append((progress, message)),
    )

    assert final_video is not None
    assert_seven_stages_in_order(updates)


def test_api_job_reports_real_stages_without_sleeping(env, monkeypatch):
    calls = []
    original_update = app_module.update_job_status

    def recording_update(job_id, status, progress=None, message=None, video_path=None):
        calls.append((status, progress, message))
        original_update(job_id, status, progress, message, video_path)

    monkeypatch.setattr(app_module, "update_job_status", recording_update)

    run_api_job(env, "job-a")

    assert calls[0] == ("processing", 0, "Initializing...")
    assert calls[-1][:2] == ("completed", 100)
    stage_updates = [(p, m) for s, p, m in calls[1:-1]]
    assert all(s == "processing" for s, _, _ in calls[1:-1])
    assert_seven_stages_in_order(stage_updates)
    assert app_module.jobs["job-a"]["status"] == "completed"


def run_api_job(env, job_id):
    output_dir = env.output_root / job_id
    output_dir.mkdir()
    app_module.jobs[job_id] = {"status": "queued", "progress": 0}
    app_module.generate_video_task(job_id, f"Ad for {job_id}", "Modern Cool", output_dir)


def download(job_id):
    response = app_module.app.test_client().get(f"/api/download/{job_id}")
    data = response.data
    response.close()
    return response.status_code, data


def test_download_serves_each_jobs_own_video(env):
    run_api_job(env, "job-a")
    run_api_job(env, "job-b")

    for job_id in ("job-a", "job-b"):
        status, data = download(job_id)
        assert status == 200
        # The fake FFmpeg writes its output path into the file
        assert str(env.output_root / job_id / "final_video_with_audio.mp4").encode() in data


def test_download_refuses_a_file_outside_the_jobs_folder(env):
    other_video = env.output_root / "job-b" / "final_video_with_audio.mp4"
    other_video.parent.mkdir()
    other_video.write_text("job-b's video")
    app_module.jobs["job-a"] = {"status": "completed", "video_path": str(other_video)}

    status, _ = download("job-a")

    assert status == 404
