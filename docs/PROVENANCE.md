# Provenance of performance numbers

This file is the evidence for every performance number the README uses. Each
number below is copied from one of these sources, and the source is named.

**Committed evidence, in `docs/runs/`:**

| File | Contents |
|---|---|
| `runs/runA-measurements.txt`, `runs/runB-measurements.txt` | Every `[Job]`, `[Timing]` and `[VRAM]` line from each run's log, copied verbatim. Lines containing a local path would have been left out; none did. |
| `runs/runA-nvsmi.csv`, `runs/runB-nvsmi.csv` | `nvidia-smi` memory samples from the same runs, copied unchanged. They contain only timestamps and MiB values. |

`docs/runs/` also holds `runs/runC-measurements.txt`, copied the same way from
Run C, a verification run with a different prompt whose numbers are not used
in the README's results.

Citations use `file:line`, e.g. `runs/runA-measurements.txt:22`.

**Full logs.** The two full run logs, runA.log and runB.log, contain local
paths, so they are not committed. They are cited only for numbers the excerpts
don't cover, and those citations are marked "(full log, not committed)".

**Code.** Code in this repository is cited as `file:line`.

Anything that cannot be traced to one of these sources is marked
**CAN'T VERIFY**.

---

## 1. How measurement works

All of this is in `src/core/measurement.py`.

- **Stages.** Each pipeline stage is wrapped in a `Stage`. When the stage stops,
  it logs `[Timing] <name>: <s> seconds` (measurement.py:86). The time is
  wall-clock time from `time.perf_counter()` (measurement.py:78, 82).
- **Per-stage peak VRAM.** When a stage starts, it calls
  `torch.cuda.reset_peak_memory_stats()` (measurement.py:76). When it stops, it
  reads `torch.cuda.max_memory_allocated()` and
  `torch.cuda.max_memory_reserved()` (measurement.py:48-49). It logs both as
  `[VRAM] <name>: peak allocated … GB, peak reserved … GB` (measurement.py:95-98).
  Resetting sets the peak to the memory in use at that moment. So a stage's peak
  can never be lower than what was already held when the stage started,
  including memory left from earlier stages (see section 4).
- **Nesting.** `SDXL Loading` runs inside `Image Generation`
  (video_generation_pipeline.py:149, 160). Before a nested stage resets the
  counters, the parent stage keeps the peak it has seen so far
  (measurement.py:73-75, 91-92). This means the time and peak for
  `Image Generation` include `SDXL Loading`. Adding up the stage times
  therefore counts SDXL loading twice.
- **Job totals.** `JobMeasurement` starts before the pipeline is built
  (app.py:79) and finishes in a `finally` (app.py:108). It logs
  `[Timing] JOB TOTAL (incl. Gemma loading)` and `[VRAM] JOB PEAK (max over
  stages)` (measurement.py:141-146). The job peak is the largest value recorded
  by any stage (measurement.py:118-120).
- **Pipeline total.** `[Timing] TOTAL PIPELINE TIME` is a separate timer.
  It starts inside `generate_video()` (video_generation_pipeline.py:62, 427-428).
  Gemma is loaded while the pipeline is being built, before this point. In both
  logs, `Gemma Loading` finishes before `STARTING VIDEO GENERATION PIPELINE`
  (runA.log:78, 88; runB.log:80, 90; full log, not committed). So the pipeline
  total does not include Gemma loading. The job total does.
- **Units.** Values are divided by `GB = 1024 ** 3` (measurement.py:16).
  They are therefore **GiB**, even though the logs print "GB". The same divisor
  is used by the other VRAM log lines (model_manager.py:106-108,
  image_generator.py:132-133, 488). `nvidia-smi` reports **MiB**, as its CSV
  header says (`memory.used [MiB]`, `runs/runA-nvsmi.csv:1`).
- **What the PyTorch counters exclude.** They track only memory that goes
  through PyTorch's CUDA caching allocator. They do not include the CUDA
  context, memory allocated outside PyTorch, or other processes. "Reserved" is
  memory the allocator holds from the driver. "Allocated" is the part of it
  that live tensors are using.
- **nvidia-smi sampling.** Consecutive CSV timestamps are about 5 s apart
  (e.g. `22:20:43.346`, `22:20:48.361`, `22:20:53.376` at
  `runs/runA-nvsmi.csv:2-4`). A memory peak shorter than one interval can be
  missed. The command used to collect the samples is not recorded:
  **CAN'T VERIFY**.

## 2. Run setup

| | Run A | Run B |
|---|---|---|
| Commit | `cfb5ca9` (before the unload fix `7fe73be`) | `934bd7c` (after it) |
| Date (log timestamps) | 2026-11-02 | 2026-11-03 |
| Job ID | `575bb334-21aa-4710-b0c0-1e84bc812443` (`runs/runA-measurements.txt:1`) | `3921645c-ccb7-4bbe-914c-91021f15dd09` (`runs/runB-measurements.txt:1`) |
| GPU | NVIDIA GeForce RTX 2050, total VRAM 4.00 GB (`runs/runA-measurements.txt:2`) | same (`runs/runB-measurements.txt:2`) |
| torch | 2.11.0+cu128, CUDA build 12.8 (`runs/runA-measurements.txt:3`) | same (`runs/runB-measurements.txt:3`) |
| Config | scenes=3, image=512x512, steps=6, video=512x512 @ 24fps (`runs/runA-measurements.txt:4`) | same (`runs/runB-measurements.txt:4`) |
| Theme | Modern Cool (runA.log:91, full log, not committed) | Modern Cool (runB.log:93, full log, not committed) |

- **Commits.** The logs do not record which commit was checked out. The
  commits above were reported by the person who ran the tests. The full logs
  (not committed) agree with them:
  - Run A logs the unload failure that `7fe73be` fixed (runA.log:145).
  - Run A logs the false "Less than 4GB VRAM" error that `d6fe8ee` fixed
    (runA.log:13).
  - Run B logs neither of these. It writes into a per-job output folder, which
    `f2a15f5` added (runB.log:202).
- **Other changes between the runs.** `934bd7c` is nine commits after `cfb5ca9`
  (`git log cfb5ca9..934bd7c`), not just the unload fix. Some of them change
  what the models are asked to produce:
  - `6aee9fb` changes the scene prompt's example.
  - `b89a364` passes the storyline's visual anchors to image generation.
  - `08ad0d5` pins dependency versions. torch is the same in both runs.
  - `f2a15f5`, `d6fe8ee` and `3ee6384` change output folders, the VRAM
    warning and progress reporting.

  So differences between the runs cannot all be put down to the unload fix.
- **Later commit.** `b8b2c48` (overlay text wrapping) came after Run B. It
  changes only how text is drawn, and neither run includes it.
- **Prompt.** Both full logs show the same prompt up to the point where the log
  cuts it off: "Create a marketing video for an AI-powered productivity app
  designed for students. Highlight distrac..." (runA.log:90, runB.log:92; full
  log, not committed). Whether the full prompt was identical: **CAN'T VERIFY**
  from the logs.
- **OS and Python.** Windows 11 and Python 3.13 are not recorded in the logs.
  The repo's venv is Python 3.13.7 (`venv/pyvenv.cfg`, not committed). That the
  runs used this venv: **CAN'T VERIFY** from the logs.
- **One run each.** With one run per commit, there is no measure of how much
  times vary from run to run (see section 6).

## 3. Per-stage measurements

Copied from the `[Timing]` and `[VRAM]` lines. VRAM is in GiB (printed as "GB").

| Stage | Run A time (s) | Run A peak alloc | Run A peak reserved | Run B time (s) | Run B peak alloc | Run B peak reserved |
|---|---|---|---|---|---|---|
| Gemma Loading | 27.90 | 2.11 | 2.12 | 20.05 | 2.11 | 2.12 |
| LLM + Scene Structuring | 239.24 | 2.23 | 2.35 | 201.40 | 2.21 | 2.33 |
| SDXL Loading (inside Image Generation) | 10.19 | 2.08 | 2.35 | 6.95 | 0.01 | 2.04 |
| Image Generation | 292.54 | 3.03 | 3.07 | 353.21 | 2.08* | 2.33* |
| Voiceover Generation | 4.79 | 2.08 | 2.12 | 3.67 | 0.01 | 2.04 |
| Audio Merge | 0.19 | 2.08 | 2.12 | 0.29 | 0.01 | 2.04 |
| Scene Video Generation | 24.87 | 2.08 | 2.12 | 19.55 | 0.01 | 2.04 |
| Final Assembly | 1.29 | 2.08 | 2.12 | 1.06 | 0.01 | 2.04 |

\* This peak was set at the start of the stage, while Gemma was still loaded.
It is not SDXL's peak. See section 4.

Sources: `runs/runA-measurements.txt:5-20` and `runs/runB-measurements.txt:5-20`.
The lines are in stage order, with a `[Timing]` line followed by a `[VRAM]`
line for each stage.

| Total | Run A | Run B |
|---|---|---|
| JOB TOTAL (incl. Gemma loading) | 594.16 s (`runs/runA-measurements.txt:22`) | 599.53 s (`runs/runB-measurements.txt:22`) |
| TOTAL PIPELINE TIME (excl. Gemma loading) | 9.38 minutes (`runs/runA-measurements.txt:21`) | 9.65 minutes (`runs/runB-measurements.txt:21`) |
| JOB PEAK allocated / reserved | 3.03 / 3.07 GiB (`runs/runA-measurements.txt:23`) | 2.21 / 2.33 GiB (`runs/runB-measurements.txt:23`) |
| Video duration | 21.77s (runA.log:303, full log, not committed) | 16.49s (runB.log:313, full log, not committed) |
| Per-scene durations | 3.55s, 6.70s, 11.52s (runA.log:211-213, full log, not committed) | 2.11s, 8.66s, 5.71s (runB.log:220-222, full log, not committed) |

Both totals are for the whole job with a single prompt, on the setup in
section 2.

## 4. Unload evidence

**Run A did not unload Gemma.** Before image generation, the pipeline tried to
unload the text model, and that failed:

```
⚠️ Text model unload warning: property 'model' of 'ModelManager' object has no deleter
```

(runA.log:145, full log, not committed). After the unload fix, Run B logs
`🗑️ Gemma unloaded` and `✅ Text models unloaded` at the same point
(runB.log:152-153, full log, not committed).

**PyTorch counters.** In Run A, Gemma's weights stay allocated through every
stage. In Run B they are gone from `SDXL Loading` onward:

| | Run A | Run B |
|---|---|---|
| Allocated right after Gemma loaded | 2.07 GB (runA.log:74, full log, not committed) | 2.07 GB (runB.log:76, full log, not committed) |
| SDXL Loading peak allocated (`runs/run*-measurements.txt:10`) | 2.08 | 0.01 |
| SDXL initial VRAM (current, not peak) | 2.08GB allocated, 2.12GB reserved (runA.log:154, full log, not committed) | 0.01GB allocated, 2.04GB reserved (runB.log:162, full log, not committed) |
| Image Generation peak allocated / reserved (`runs/run*-measurements.txt:12`) | 3.03 / 3.07 | 2.08 / 2.33 (set before the unload, see below) |
| Voiceover → Final Assembly peak allocated (`runs/run*-measurements.txt:14, 16, 18, 20`) | 2.08 in every stage | 0.01 in every stage |
| JOB PEAK allocated / reserved (`runs/run*-measurements.txt:23`) | 3.03 / 3.07 | 2.21 / 2.33 |

**Where the Image Generation peaks come from.** The `Image Generation` stage
starts (video_generation_pipeline.py:149) before the text model is unloaded
(video_generation_pipeline.py:156). Its peak counters are reset while Gemma is
still loaded.

- **Run A.** The unload failed, so Gemma stayed loaded while SDXL ran. The
  stage's 3.03 / 3.07 GiB peak (`runs/runA-measurements.txt:12`) is
  Gemma and SDXL together. This is the job peak.
- **Run B.** The stage started right after `LLM + Scene Structuring` stopped at
  20:10:42,919 (`runs/runB-measurements.txt:8`). The unload finished at
  20:10:45,168 (runB.log:152, full log, not committed). The stage's peak,
  2.08 allocated / 2.33 reserved (`runs/runB-measurements.txt:12`), was set at
  the start of the stage, with Gemma still loaded:
  - 2.08 GiB is what Gemma held with nothing else allocated. Compare 2.07 GB
    right after loading (runB.log:76, full log, not committed) and 2.08 in
    every Run A stage after the LLM (`runs/runA-measurements.txt:10, 14`).
  - 2.33 GiB equals Run B's LLM-stage peak reserved
    (`runs/runB-measurements.txt:8`).

  Once Gemma was unloaded, `SDXL Loading` peaked at 0.01 allocated / 2.04
  reserved (`runs/runB-measurements.txt:10`).
- **SDXL's own peak is not measured separately.** The counters are not reset
  between SDXL loading and generation. So all they show is that SDXL's
  generation peak was at most 2.08 GiB allocated and 2.33 GiB reserved, the
  stage's peak. Its actual value: **CAN'T VERIFY**.
- **The per-image "VRAM used" lines are not peaks.** Run B logs
  `💾 VRAM used: 0.01GB` after each image (runB.log:176, 182, 188, full log,
  not committed). These lines are point readings of
  `torch.cuda.memory_allocated()` taken after each image finishes
  (image_generator.py:488-489). They do not show the peak during generation.
- **Run B's job peak.** The job peak, 2.21 / 2.33 GiB, comes from the
  `LLM + Scene Structuring` stage (`runs/runB-measurements.txt:8, 23`).

**nvidia-smi before the job.** Both CSVs read 0 MiB until the job started:

- `runs/runA-nvsmi.csv:2-35`: 34 samples, `22:20:43.346`–`22:23:28.658`
- `runs/runB-nvsmi.csv:2-81`: 80 samples, `20:00:22.606`–`20:06:58.350`

**nvidia-smi after the job.** Memory stayed held until sampling ended:

| | Run A | Run B |
|---|---|---|
| Value after the job | 2273 MiB | 2193 MiB |
| CSV lines | `runs/runA-nvsmi.csv:147-428` | `runs/runB-nvsmi.csv:196-257` |
| Samples | 282 | 62 |
| Time span | `22:32:49.460`–`22:56:16.704` | `20:16:34.258`–`20:21:39.672` |

The jobs ended at 22:33:22,443 (`runs/runA-measurements.txt:22`) and
20:17:00,719 (`runs/runB-measurements.txt:22`). 2273 MiB = 2.2197 GiB and
2193 MiB = 2.1416 GiB (divided by 1024). The two runs hold this memory for
different reasons:

- **Run B: PyTorch's cache.** After the unload, allocated memory is 0.01 GiB
  but reserved memory is 2.04 GiB (`runs/runB-measurements.txt:10, 14, 16, 18, 20`;
  runB.log:162, full log, not committed). The caching allocator keeps this
  memory reserved without any tensor using it, and `nvidia-smi` counts it as
  used. `ModelManager.unload()` calls `torch.cuda.empty_cache()`
  (model_manager.py:228), and the pipeline calls it again during cleanup
  (video_generation_pipeline.py:483). Even so, reserved stayed at 2.04 GiB.
  Why `empty_cache()` did not hand this memory back to the driver:
  **CAN'T VERIFY**.
- **Run A: a loaded model, not just cache.** Allocated memory is still
  2.08 GiB at the end of Final Assembly (`runs/runA-measurements.txt:20`).
  Those are live tensors, so the idle memory in Run A is mostly Gemma itself,
  still loaded, not free cache. No PyTorch reading was taken after the job
  ended, so the allocated value during the idle period itself: **CAN'T VERIFY**.
- **The remainder.** Each run's idle `nvidia-smi` value is above its last
  reserved value: 2.12 GiB in Run A, 2.04 GiB in Run B
  (`runs/run*-measurements.txt:20`). This is consistent with memory outside
  PyTorch, such as the CUDA context. The size of this remainder:
  **CAN'T VERIFY**. The logged values are rounded to 0.01 GiB, so they cannot
  be subtracted precisely.

**A missed peak in Run A.** Run A's Image Generation stage reports peak
reserved 3.07 GiB (`runs/runA-measurements.txt:12`), which is 3143.7 MiB.
The highest value in the whole of `runs/runA-nvsmi.csv` is 2657 MiB, at
`22:29:44.184` (`runs/runA-nvsmi.csv:110`). `nvidia-smi` never showed a peak
that PyTorch recorded. This fits the 5 s sampling missing short peaks. Whether
any other effect also contributes, such as how Windows reports GPU memory:
**CAN'T VERIFY**.

**Run B shows no missed peak.** Its 2.33 GiB peak was set at the start of the
image stage, before the unload (see above). A sample taken in that window
reads 2479 MiB at `20:10:43.658` (`runs/runB-nvsmi.csv:126`), which is above
2.33 GiB (2385.9 MiB).

Use the PyTorch counters for peaks and `nvidia-smi` for sustained levels.

## 5. Old README claims compared with the measurements

The old numbers come from `README.md` before this change. Which commit,
hardware state or prompt produced them is not recorded: **CAN'T VERIFY**.
Measured values are from section 3 and its sources. Sums in the table add
values from section 3.

| Old claim (README.md line) | Run A | Run B | Verdict |
|---|---|---|---|
| LLM + Scene Structuring "~98 seconds" (112), "98 seconds" (284) | 239.24 s | 201.40 s | Not reproduced. Both runs took more than twice as long. |
| Image Generation "~308 seconds" (113), "308 seconds" (285) | 292.54 s | 353.21 s | Between the two runs. One run each, so it is neither confirmed nor refuted. |
| Voiceover + Audio Merge "~5 seconds" (114) | 4.79 + 0.19 = 4.98 s | 3.67 + 0.29 = 3.96 s | Consistent with Run A. Run B was lower. |
| Scene Video Rendering "~29 seconds" (115), "28 seconds" (286) | 24.87 s | 19.55 s | Not reproduced. Both runs were lower. The README also gives two different values for the same stage. |
| Final Assembly "~1.4 seconds" (116), "1.4 seconds" (287) | 1.29 s | 1.06 s | Not reproduced. Both runs were lower. |
| Total Pipeline Time "~7 minutes" (117) | 9.38 min pipeline / 594.16 s job | 9.65 min pipeline / 599.53 s job | Not reproduced. Whether "~7 minutes" included Gemma loading: CAN'T VERIFY. |
| "~23s video" (108) | 21.77s | 16.49s | Not reproduced. Video length follows the voiceover, so it changes from run to run. |
| "20–30 seconds video" (233) | 21.77s | 16.49s | Run B is outside this range. |
| Peak VRAM "~2.0–2.2GB active allocation" (102) | job peak allocated 3.03 GiB | job peak allocated 2.21 GiB | Not reproduced before the unload fix (3.03). Just above the range after it (2.21). |
| "stable under the 4GB limit" (102) | peak reserved 3.07 GiB of 4.00 GB | peak reserved 2.33 GiB of 4.00 GB | Both runs finished, and both peaks stayed under the total. Two runs do not establish stability in general. |
| "Performance scales approximately linearly with scene count" (119) | 3 scenes | 3 scenes | CAN'T VERIFY. No run used a different scene count. |
| "Full pipelines can take 20+ minutes" (318) | — | — | CAN'T VERIFY. No log of an unoptimised pipeline exists. |

## 6. Variation between the runs

There is one run per commit, so these differences are reported as they were
measured. They are not averaged and not used to estimate how much times vary.
Stage times are from section 3. The intervals below are worked out from
timestamps in the full logs (not committed).

**LLM + Scene Structuring: 239.24 s vs 201.40 s** (`runs/run*-measurements.txt:7`).

| Part | Run A | Run B |
|---|---|---|
| Storyline: "Generating storyline..." → "Storyline generated successfully" | 22:23:59,509 → 22:26:08,782 = 129.273 s | 20:07:21,516 → 20:08:46,165 = 84.649 s |
| Storyline attempts | one attempt, 129.270 s to the response (runA.log:99-101) | 4.079 s to a non-JSON response, then a retry (runB.log:101-106); attempt 2 took 78.554 s (runB.log:106-107) |
| Scene descriptions: "Generating scenes sequentially" → "Generated 3 scenes" | 109.932 s | 116.736 s |

- Run B's JSON retry happened: attempt 1 returned a code block instead of
  JSON, and the log shows `⚠️ Could not extract JSON on attempt 1`
  (runB.log:105).
- The retry cost 6.082 s from the start of attempt 1 to the start of attempt 2
  (runB.log:101, 106).
- Even so, Run B's stage was the faster one. The difference comes from Run A's
  single storyline response taking 129.270 s, against 78.554 s for Run B's
  successful attempt. Why it took longer is not logged: the storyline
  response's length is not logged. The scene-prompt changes between the
  commits (section 2) may also play a part: **CAN'T VERIFY**.

**Image Generation: Run B's first image.** Per image, from "Generating cinematic
image for Scene N..." to "Progress: N/3 images generated":

| Image | Run A | Run B |
|---|---|---|
| 1 | 98.396 s | 164.186 s |
| 2 | 92.284 s | 87.183 s |
| 3 | 90.426 s | 91.536 s |

Measured to "Scene N image saved" instead, Run B's first image takes 162.641 s.
The others take 85.312–96.664 s.

Run B's first image accounts for most of the 60.67 s gap between the two Image
Generation stages (353.21 − 292.54, `runs/run*-measurements.txt:11`). What
caused it is not logged: **CAN'T VERIFY**. Run B's image prompts also differ
from Run A's because of the visual-anchor change between the commits
(section 2).

**Other differences.**
- Gemma Loading: 27.90 s vs 20.05 s (`runs/run*-measurements.txt:5`).
- SDXL Loading: 10.19 s vs 6.95 s (`runs/run*-measurements.txt:9`).

The cause of either difference is not logged: **CAN'T VERIFY**.
