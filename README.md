# AI Video Marketing Ad Generator

This project is a fully local, resource-aware AI pipeline that transforms a single text prompt into a complete marketing video — including script, visuals, voiceover, motion effects, and final rendering.

It is specifically engineered to run on low-VRAM hardware (4GB GPU) using quantization and staged model loading.

---

## What This Project Demonstrates

This is a **memory-constrained AI systems engineering project** focused on:

- Running multiple generative models on a 4GB GPU
- 4-bit LLM quantization with FP16 compute
- Sequential model unloading to manage VRAM
- CPU offloading for diffusion inference
- Stage-level runtime instrumentation and logging
- End-to-end automation from prompt → final MP4

---

## System Architecture

```
User Prompt
→ Flask REST API
→ VideoGenerationPipeline
→ Storyline & Scene Structuring (Gemma 2B IT)
→ Scene Normalization
→ Text Model Unload
→ Image Generation (SDXL Turbo 512x512)
→ Diffusion Model Unload
→ Voiceover Generation
→ Scene Video Composition
→ FFmpeg Stitching
→ Final Video (MP4)
```

---

## Default Runtime Configuration

| Component | Value |
|---------|------|
Scenes | 3
Resolution | 512x512
Diffusion Steps | 6
Video FPS | 24
Upscaling | Disabled

The architecture supports a **5-scene marketing structure**, but runtime uses **3 scenes** for 4GB GPUs.

---

## Models Used

### Text Model

Gemma 2 2B IT

- 4-bit quantization (BitsAndBytes)
- FP16 compute
- Local loading

### Image Model

SDXL Turbo

- 512x512 inference
- CPU offload enabled
- Attention slicing enabled

Models stored locally:

```
models/
 ├── gemma-2-2b-it/
 └── sdxl-turbo/
```

Models load using:

```
local_files_only=True
```

---

## Hardware Compatibility

Tested on:

- GPU: RTX 2050
- VRAM: 4GB
- RAM: 8GB

Peak VRAM:

```
~2.0–2.2GB
```

---

## Performance

Example run:

| Stage | Time |
|------|------|
LLM + Scenes | ~98s
Images | ~308s
Voice | ~5s
Video | ~29s
Final | ~1.4s

Total:

```
~7 minutes
```

---

## Installation

Clone:

```
git clone https://github.com/Jeswin987/ai-video-ad-generator.git
cd ai-video-ad-generator
```

Create environment:

```
python -m venv venv
venv\Scripts\activate
```

Install:

```
pip install -r requirements.txt
```

---

## Model Setup

Download models and place into:

```
models/
 ├── gemma-2-2b-it/
 └── sdxl-turbo/
```

Example:

```
models/gemma-2-2b-it/config.json
models/sdxl-turbo/model_index.json
```

---

## Run Server

```
python app.py
```

Open:

```
http://localhost:5000
```

---

## Project Structure

```
ai-video-ad-generator/

app.py
requirements.txt

src/
 core/
 generation/
 media/

models/
output/
logs/
temp/
```

---

## Logging

Example:

```
[Timing] LLM + Scene Structuring: 98 seconds
[Timing] Image Generation: 308 seconds
[Timing] Scene Video Generation: 28 seconds
[Timing] Final Assembly: 1.4 seconds
```

Logs stored in:

```
logs/
```

---

## License

Educational and research purposes.