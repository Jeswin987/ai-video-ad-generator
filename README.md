# 🎬 AI Video Marketing Ad Generator

> A fully local, resource-aware AI pipeline that transforms a single text prompt into a complete marketing video — including script, visuals, voiceover, motion effects, and final rendering.

Specifically engineered to run on **low-VRAM hardware (4GB GPU)** using quantization and staged model loading.

---

## 🧠 What This Project Demonstrates

This is a **memory-constrained AI systems engineering project** focused on:

- Running multiple generative models on a 4GB GPU
- 4-bit LLM quantization with FP16 compute
- Sequential model unloading to manage VRAM
- CPU offloading for diffusion inference
- Stage-level runtime instrumentation and logging
- End-to-end automation from prompt → final MP4

---

## 🏗️ System Architecture (Runtime Flow)

```
User Prompt
  → Flask REST API
  → VideoGenerationPipeline (Core Orchestrator)
  → Storyline & Scene Structuring (Gemma 2 2B IT – 4-bit quantized)
  → Scene Normalization
  → Text Model Unload (VRAM release)
  → Image Generation per Scene (SDXL Turbo 512x512)
  → Diffusion Model Unload
  → Voiceover Generation + Duration Sync
  → Scene Video Composition (Ken Burns motion)
  → Media Stitching (FFmpeg)
  → Final Rendered Video (MP4)
```

---

## ⚙️ Default Runtime Configuration (4GB Optimized Mode)

| Component        | Value    |
|------------------|----------|
| Scenes           | 3        |
| Resolution       | 512x512  |
| Diffusion Steps  | 6        |
| Upscaling        | Disabled |
| Video FPS        | 24       |

> The conceptual architecture supports a **5-scene marketing structure**, but runtime uses **3 scenes** to ensure VRAM stability on 4GB GPUs.

---

## 🤖 Models Used

### Text Model — Gemma 2 2B IT

- Fully local loading
- `load_in_4bit=True` using BitsAndBytes
- FP16 compute dtype
- Full-precision weights stored on disk
- Quantized dynamically at load time

### Image Model — SDXL Turbo

- 512x512 inference
- Sequential CPU offload
- Attention slicing enabled
- VAE optimizations enabled

---

## 💾 Model Storage

Models are downloaded once from Hugging Face and stored locally:

```
models/
├── gemma-2-2b-it/
└── sdxl-turbo/
```

Runtime loading uses `local_files_only=True`, which guarantees:

- ✅ No runtime downloads
- ✅ Fully offline operation
- ✅ Faster startup

---

## 🖥️ Hardware Compatibility

Tested on:

| Component | Spec          |
|-----------|---------------|
| GPU       | NVIDIA RTX 2050 |
| VRAM      | 4GB           |
| RAM       | 8GB           |

**Peak VRAM usage:** `~2.0–2.2GB` active allocation — stable under the 4GB limit.

---

## 📊 Performance Benchmarks (RTX 2050)

Example run (3 scenes, ~23s video):

| Stage                    | Time          |
|--------------------------|---------------|
| LLM + Scene Structuring  | ~98 seconds   |
| Image Generation         | ~308 seconds  |
| Voiceover + Audio Merge  | ~5 seconds    |
| Scene Video Rendering    | ~29 seconds   |
| Final Assembly           | ~1.4 seconds  |
| **Total Pipeline Time**  | **~7 minutes** |

> Performance scales approximately linearly with scene count.

---

## 📦 Output

Each generation produces:

- AI-generated scene images
- Voiceover audio files
- Per-scene video clips
- Final MP4 with audio
- Scene metadata JSON
- Runtime logs

```
output/
├── images/
│   ├── scene_1_image.png
│   ├── scene_2_image.png
│   └── scene_3_image.png
├── audio/
│   ├── audio_scene_1.mp3
│   ├── audio_scene_2.mp3
│   └── audio_scene_3.mp3
└── final_video_with_audio.mp4
```

---

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/Jeswin987/ai-video-ad-generator.git
cd ai-video-ad-generator
```

**2. Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 📥 Model Setup

Download models from Hugging Face and place them into the `models/` directory:

```
models/
├── gemma-2-2b-it/
│   └── config.json
└── sdxl-turbo/
    └── model_index.json
```

The system loads models strictly from local storage using `local_files_only=True`, ensuring:

- No runtime downloads
- Offline operation
- Deterministic execution

---

## ▶️ Running the Server

```bash
python app.py
```

Server runs at: `http://localhost:5000`

Expected startup logs:

```
AI VIDEO GENERATION API SERVER
Device: cuda
Image Generation: Enabled
Pipeline ready
```

---

## 🌐 Web Interface

Open `http://localhost:5000` in your browser.

Features:
- Prompt input
- Video generation trigger
- Progress monitoring
- Video download

---

## 💡 Example Prompt

```
Create a marketing video for an AI-powered productivity app designed for students.
Highlight distraction problems and show how the AI assistant improves focus.
```

Typical output:
- 3 scenes
- 20–30 seconds video
- AI-generated visuals
- Voiceover narration
- Ken Burns animated motion
- Final MP4 render

---

## 🗂️ Project Structure

```
ai-video-ad-generator/
├── app.py
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── model_manager.py
│   │   └── video_generation_pipeline.py
│   ├── generation/
│   │   ├── storyline_generator.py
│   │   ├── scene_descriptor.py
│   │   └── scene_normalizer.py
│   └── media/
│       ├── image_generator.py
│       ├── voiceover_generator.py
│       ├── audio_merger.py
│       ├── video_generator_with_images.py
│       └── media_merger.py
├── models/
├── output/
├── logs/
└── temp/
```

---

## 📋 Logging and Instrumentation

The pipeline includes detailed runtime instrumentation. Each run records:

- Scene generation time
- Diffusion time
- Audio generation time
- Video rendering time
- Final assembly time
- Total runtime

Example log output:

```
[Timing] LLM + Scene Structuring:  98 seconds
[Timing] Image Generation:        308 seconds
[Timing] Scene Video Generation:   28 seconds
[Timing] Final Assembly:            1.4 seconds
```

Logs are stored in the `logs/` directory.

---

## 🔧 Engineering Challenges Solved

### VRAM Constraints

**Problem:** Running LLM and diffusion models together exceeds 4GB VRAM.

**Solution:**
- 4-bit LLM quantization
- Sequential model unloading
- CPU offloading
- FP16 compute

### Memory Stability

**Problem:** Diffusion models crash on low-VRAM GPUs.

**Solution:**
- CPU offload pipeline
- Reduced resolution (512x512)
- Reduced diffusion steps (6)
- Controlled pipeline execution order

### Runtime Optimization

**Problem:** Full pipelines can take 20+ minutes.

**Solution:**
- SDXL Turbo for fast inference
- Low-step diffusion
- Optimized scene count
- Efficient FFmpeg media assembly

---

## 🔮 Future Improvements

- [ ] Higher resolution generation
- [ ] 5-scene runtime mode
- [ ] Better voice synthesis models
- [ ] Faster diffusion schedulers
- [ ] GPU batching support
- [ ] Style templates

---

## 📄 License

For educational and research purposes.