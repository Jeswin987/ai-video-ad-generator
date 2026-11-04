# AI Video Ad Generator

A pipeline that turns a text prompt into a short marketing video on a 4 GB
laptop GPU. Everything runs locally except the voiceover, which uses the online
gTTS service. Gemma 2 2B IT, loaded in 4-bit, writes a storyline and turns
it into scenes. SDXL-Turbo draws one 512x512 image per scene. gTTS reads each
scene's voiceover. Each image becomes a Ken Burns
(slow zoom and pan) scene video with a text overlay, and FFmpeg joins the
scenes and the audio into one MP4. A Flask server runs the pipeline and serves
a small web UI.

All numbers below come from [docs/PROVENANCE.md](docs/PROVENANCE.md), which
cites the log line or code line behind each one.

## Results at a glance

One run (Run B) on an NVIDIA GeForce RTX 2050 (4 GB), 3 scenes, 512x512
images, 6 steps, 512x512 video at 24 fps:

| Stage | Time (s) |
|---|---|
| Gemma Loading | 20.05 |
| LLM + Scene Structuring | 201.40 |
| Image Generation (includes SDXL Loading, 6.95 s) | 353.21 |
| Voiceover Generation | 3.67 |
| Audio Merge | 0.29 |
| Scene Video Generation | 19.55 |
| Final Assembly | 1.06 |
| **Job total, including model loading** | **599.53** |

- **Job peak VRAM:** 2.21 GiB allocated, 2.33 GiB reserved, set during
  LLM + Scene Structuring. These are PyTorch's allocator counters: they leave
  out the CUDA context and anything allocated outside PyTorch.
- **Video length:** 16.49 s.

**Before and after the unload fix.** Before `7fe73be`, Gemma was never
actually unloaded, so it stayed on the GPU while SDXL-Turbo ran. Run A was
taken before the fix, Run B after it:

| | Run A | Run B |
|---|---|---|
| Job peak allocated | 3.03 GiB | 2.21 GiB |
| Allocated in every stage after image generation | 2.08 GiB | 0.01 GiB |

Caveats, stated once here:

- Each number is from a single run, so there is no measure of run-to-run
  variation.
- SDXL-Turbo's own peak is not measured separately (see
  [PROVENANCE.md section 4](docs/PROVENANCE.md#4-unload-evidence)).
- Run A and Run B are nine commits apart, not only the unload fix, so not
  every difference between them comes from that fix.
- Later commits change what Run B ran. `b8b2c48` and `4c72b51` change only how
  overlay text is drawn and the text of a startup log message. The web UI fix
  that came after them maps theme names to recognised themes, which changes
  the palette in the image prompts but not the timing or memory path. Both
  runs sent "Modern Cool", which at the time matched no theme and got the
  default palette.

## How it fits in 4 GB

- **4-bit Gemma.** `src/core/model_manager.py` loads Gemma with
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
  bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")` and
  `device_map="cuda"`. The weights are 16-bit (bf16) on disk, quantized to
  4-bit NF4 at load time.
- **Sequential CPU offload for SDXL-Turbo.** `src/media/image_generator.py`
  loads the fp16 variant and calls `enable_sequential_cpu_offload()`, so
  layers move to the GPU only while they run. It also enables attention
  slicing, VAE slicing and VAE tiling.
- **One model at a time.** The pipeline unloads Gemma before it loads
  SDXL-Turbo, and unloads SDXL-Turbo before the voiceover stage. Each job
  loads Gemma again.

## Quick start

Tested on Windows 11 with Git Bash, Python 3.13 and an NVIDIA GPU.

Prerequisites:

- FFmpeg and ffprobe on `PATH`
- A Hugging Face account that has accepted the Gemma licence
- Disk space, measured with `du -sh` on the test machine: about 11.4 GiB for
  the models (`gemma-2-2b-it` 4.9 GiB + `sdxl-turbo` 6.5 GiB), plus 5.0 GiB
  for the venv
- Internet access while generating, for gTTS

```bash
git clone https://github.com/Jeswin987/ai-video-ad-generator.git
cd ai-video-ad-generator
py -3.13 -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
hf auth login
hf download google/gemma-2-2b-it --revision 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8 --local-dir models/gemma-2-2b-it --include "*.json" --include "*.safetensors" --include "tokenizer.model"
hf download stabilityai/sdxl-turbo --revision 71153311d3dbb46851df1931d3ca6e939de83304 --local-dir models/sdxl-turbo --include "*.json" --include "*.txt" --include "*.fp16.safetensors"
python app.py
```

Install torch first, from the PyTorch index: the torch package on PyPI is
CPU-only on Windows. In the `hf download` commands, use one `--include` per
pattern: the hf CLI ignores extra patterns after a single flag. Models are
loaded with `local_files_only=True`, so nothing is downloaded at run time.

## Using it

### Web UI

Open http://127.0.0.1:5000. The page checks the server's health, then takes a
product description and a visual theme (one per recognised theme, Modern
Cool first).
**Generate Video** starts a job; the page polls its status every 2 seconds,
shows the current stage and percentage, and plays the finished video in the
page. **Download Video** downloads it from `/api/download/<job_id>`. **Create
New** and, after a failure, **Try Again** return to the input form and keep the
prompt text. The header shows "Server not reachable" when an API call gets no
response, and switches back to connected on any response, including an
error. Page load, Create New and Try Again also run a health check. Jobs are
kept in server memory, so after a server restart the page stops polling and
shows "Job not found. The server may have restarted." There is no cancel: a
job runs to the end once it has started.

### API

```bash
# Start a job. Returns 202 with a job_id.
curl -X POST http://127.0.0.1:5000/api/generate-video \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a marketing video for an AI-powered productivity app designed for students.", "theme": "cool"}'

# Status: status, progress, stage/message, video_ready, video_url, error
curl http://127.0.0.1:5000/api/status/<job_id>

# Download the finished MP4
curl -o video.mp4 http://127.0.0.1:5000/api/download/<job_id>
```

`theme` is optional and sets the image prompt's palette. The recognised themes
are `professional blue` (the default), `green`, `purple`, `warm` and `cool`.
The web UI's names "Modern Cool" and "Warm Professional" map to `cool` and
`warm`. Matching ignores case; an unknown name gets the default and a warning
in the log. `GET /api/health` and `GET /api/jobs` also exist.

### Progress stages

`progress` marks which stage has started, not time remaining:

| progress | message |
|---|---|
| 0 | Initializing... (Gemma loads here) |
| 12 | Stage 1/7: Generating storyline |
| 25 | Stage 2/7: Generating scene descriptions |
| 37 | Stage 3/7: Generating images |
| 50 | Stage 4/7: Generating voiceovers |
| 62 | Stage 5/7: Merging audio |
| 75 | Stage 6/7: Creating scene videos |
| 87 | Stage 7/7: Assembling final video |
| 100 | completed, or failed with an error message |

Most of the time goes to stages 1-3 (see the table above).

### LAN access

The server listens on 127.0.0.1 only. To reach it from other devices, opt in
with `APP_HOST=0.0.0.0 python app.py`; the server logs a warning when it is
reachable from the network. `scripts/setup_firewall.bat` (run as
Administrator) adds a Windows Firewall rule for port 5000. There is no
authentication.

### Output

Each job writes into its own folder:

```
output/<job_id>/
├── images/
│   ├── scene_N_image.png
│   └── scene_N_prompt.json      # prompts and settings used for the image
├── audio/
│   └── audio_scene_N.mp3
├── videos/
│   └── scene_N_video.mp4
├── merged_audio.mp3
├── concatenated_video.mp4
└── final_video_with_audio.mp4   # the file /api/download serves
```

Each server run also writes a log to `logs/run_<YYYYmmdd_HHMMSS>.log`, with
`[Timing]` and `[VRAM]` lines for every stage.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

70 tests. They need no GPU, model weights or network: a fixture in
`tests/conftest.py` makes any attempt to load a real model fail the test.

## Limitations

- **Voice.** gTTS needs internet access and sounds robotic. It has no speed
  control beyond a "slow" flag.
- **Images.** SDXL-Turbo at 512x512 often gets faces and hands wrong.
- **Steps.** 6 inference steps, above SDXL-Turbo's 1-4 step design range;
  kept from the original project.
- **Negative prompt.** Guidance scale is 0, so the negative prompt has no
  effect.
- **No seeds.** Every run gives different text, images and video.
- **Scene count.** The storyline prompt asks for 5 scenes, but only 3
  (`NUM_SCENES`) are rendered.
- **Scene length.** Each scene follows its voiceover, clamped to 2-12 s.
- **One job at a time.** There is no queue: each request starts its own
  thread, and nothing stops a second job from starting while one runs.
- **Memory after unloading.** PyTorch keeps 2.04 GiB reserved after the models
  are unloaded, even though `empty_cache()` is called.
- **Dev server.** Flask's development server, on localhost by default.
- **Hardware.** Tested only on Windows 11 with an RTX 2050 (4 GB).
- **Timings.** Single runs.

## What changed in November 2026

See the commit history for details.

- Measurement: per-stage time and peak VRAM, written to a per-run log
  (`382d9f3`).
- Unload fix: Gemma is actually unloaded before image generation (`7fe73be`).
- Per-job outputs: each job writes into its own folder (`f2a15f5`).
- Dev-server security: debugger off, localhost by default (`cfb5ca9`).
- Prompt example and visual anchors: the scene prompt's example comes from the
  storyline, and its visual anchors reach image generation (`6aee9fb`,
  `b89a364`).
- Per-scene overlays and text fitting: each scene gets its own overlay text,
  wrapped to fit the frame (`934bd7c`, `b8b2c48`).
- Real progress: the API reports actual pipeline stages instead of simulated
  progress (`3ee6384`).
- Log messages: 4 GB cards are no longer flagged as below 4 GB (`d6fe8ee`),
  and the startup time estimate uses measured values (`4c72b51`).
- Pinned dependencies (`08ad0d5`).
- Evidence: `docs/PROVENANCE.md`, plus the measurement excerpts and
  nvidia-smi samples in `docs/runs/` (`3660a22`).
- Web UI: theme names map to recognised themes, Download Video, Create New
  and Try Again work, and controls that couldn't work are removed (this
  commit).
- Tests: pytest setup with a guard against loading real models (`b7abe50`,
  `b96d608`), plus tests added in each `fix:` commit.

## Project structure

```
ai-video-ad-generator/
├── app.py                        # Flask server, API, job threads
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── measurement.py        # [Timing] / [VRAM] logging
│   │   ├── model_manager.py      # Gemma loading and unloading
│   │   └── video_generation_pipeline.py
│   ├── generation/
│   │   ├── storyline_generator.py
│   │   ├── scene_descriptor.py
│   │   └── scene_normalizer.py
│   └── media/
│       ├── image_generator.py    # SDXL-Turbo
│       ├── image_upscaler.py     # disabled by default
│       ├── voiceover_generator.py
│       ├── audio_merger.py
│       ├── video_generator_with_images.py
│       ├── easing_functions.py
│       └── media_merger.py
├── VideoGen-PWA/                 # web UI served by app.py
├── tests/
├── docs/
│   ├── PROVENANCE.md
│   └── runs/                     # measurement excerpts and nvidia-smi samples
└── scripts/
    └── setup_firewall.bat
```

`models/`, `output/`, `logs/` and `temp/` are created locally and are not in
git.

## Future improvements

- Use Gemma's chat template for prompts.
- Offline TTS.
- A job queue.
- Tune step count and image quality.
- Release PyTorch's cached memory after unloading.

## Credits

Built as a course mini-project at DUK (Sep–Dec 2025) by a three-person team:
Jesvin Devasia, Biswa Das, and Sumaya. The code was published to this
repository in February 2026, so the git history doesn't show who wrote which
part. The November 2026 polish (measurement, bug fixes, tests; see the commit
history) is by Jesvin Devasia.
