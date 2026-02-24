AI Video Marketing Ad Generator

This project is a fully local, resource-aware AI pipeline that transforms a single text prompt into a complete marketing video — including script, visuals, voiceover, motion effects, and final rendering.

It is specifically engineered to run on low-VRAM hardware (4GB GPU) using quantization and staged model loading.

What This Project Demonstrates

This is not just a “video generator.” It is a memory-constrained AI systems engineering project focused on:

Running multiple generative models on a 4GB GPU

4-bit LLM quantization with FP16 compute

Sequential model unloading to manage VRAM

CPU offloading for diffusion inference

Stage-level runtime instrumentation and logging

End-to-end automation from prompt → final MP4

System Architecture (Runtime Flow)

User Prompt
→ Flask REST API
→ VideoGenerationPipeline (Core Orchestrator)
→ Storyline & Scene Structuring (Gemma 2 2B IT – 4-bit quantized, FP16 compute)
→ Scene Normalization (Structured Scene Schema)
→ Text Model Unload (VRAM release)
→ Image Generation per Scene (SDXL Turbo @ 512x512 with CPU offload)
→ Diffusion Model Unload
→ Voiceover Generation (gTTS) + Duration Sync
→ Scene-Level Video Composition (Ken Burns motion @ 24fps)
→ Media Stitching (FFmpeg)
→ Final Rendered Marketing Video (MP4)

Default Runtime Configuration (4GB Optimized Mode)

Scenes: 3 (runtime optimized)

Resolution: 512x512

Diffusion Steps: 6

Upscaling: Disabled

Video: 24 FPS

Motion: Ken Burns effect (audio-aware easing)

The original conceptual architecture supports a 5-scene marketing structure, but the default configuration uses 3 scenes to maintain VRAM stability and reduce runtime on 4GB GPUs.

Models Used
Text Model

Gemma 2 2B IT

Loaded locally

load_in_4bit=True (BitsAndBytes)

FP16 compute dtype

Full-precision weights stored on disk

Quantized dynamically at load time

Image Model

SDXL Turbo

512x512 inference

Sequential CPU offload

Attention slicing enabled

VAE optimizations enabled

All models are downloaded once from Hugging Face and stored locally inside the models/ directory.
Runtime loading uses local_files_only=True to ensure no network calls occur during generation.

Hardware Compatibility

Tested on:

GPU: NVIDIA RTX 2050

VRAM: 4GB

System RAM: 8GB

Peak VRAM usage during generation:
~2.0–2.2GB active allocation
Stable under 4GB limit

Performance Benchmarks (RTX 2050 – 4GB Mode)

Example run (3 scenes, ~23s final video):

LLM + Scene Structuring: ~98 seconds

Image Generation: ~308 seconds

Voiceover + Audio Merge: ~5 seconds

Scene Video Rendering: ~29 seconds

Final Assembly: ~1.4 seconds

Total Pipeline Time:
~7.3 minutes

Performance scales roughly linearly with scene count due to diffusion inference cost.

Output

The system produces:

AI-generated scene images

Synchronized voiceovers

Animated scene videos

Final merged marketing video (MP4)

Prompt metadata per scene (JSON)

Detailed runtime logs per stage

Installation

Clone the repository:

git clone https://github.com/Jeswin987/ai-video-ad-generator.git
cd ai-video-ad-generator

Create a virtual environment (recommended):

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Model Setup

Models must be downloaded once and placed inside the models/ directory.

Expected structure:

models/
 ├── gemma-2-2b-it/
 └── sdxl-turbo/

The system loads models strictly from local storage.

Runtime loading uses:

local_files_only=True

This guarantees:

No runtime downloads

Fully offline operation

Faster startup times

Running the Server

Start the API server:

python app.py

Server will start at:

http://localhost:5000

You should see logs similar to:

AI VIDEO GENERATION API SERVER
Device: cuda
Image Generation: Enabled
Pipeline ready
Web Interface

Open the browser:

http://localhost:5000

The web interface allows:

Prompt input

Video generation

Progress monitoring

Video download

Example Prompt

Example marketing prompt:

Create a compelling marketing video for an AI-powered productivity app designed for ambitious students. Highlight problems like distraction and disorganization, and show how the AI assistant improves focus and productivity.

Typical output:

3 scenes

~20–30 seconds video

AI-generated visuals

Voiceover narration

Animated motion

Final MP4 render

Project Structure
ai-video-ad-generator/
│
├── app.py
├── requirements.txt
│
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── model_manager.py
│   │   └── video_generation_pipeline.py
│   │
│   ├── generation/
│   │   ├── storyline_generator.py
│   │   ├── scene_descriptor.py
│   │   └── scene_normalizer.py
│   │
│   └── media/
│       ├── image_generator.py
│       ├── voiceover_generator.py
│       ├── audio_merger.py
│       ├── video_generator_with_images.py
│       └── media_merger.py
│
├── models/
│   ├── gemma-2-2b-it/
│   └── sdxl-turbo/
│
├── output/
├── logs/
└── temp/
Logging and Instrumentation

The system includes detailed runtime logging for performance analysis.

Each generation records:

Model loading time

Scene generation time

Diffusion inference time

Voiceover generation time

Video rendering time

Final assembly time

Total pipeline runtime

Example log output:

[Timing] LLM + Scene Structuring: 98 seconds
[Timing] Image Generation: 308 seconds
[Timing] Scene Video Generation: 28 seconds
[Timing] Final Assembly: 1.4 seconds
[Timing] TOTAL PIPELINE TIME: 7.3 minutes

Logs are stored in:

logs/
Engineering Challenges Solved

This project addresses several real engineering constraints:

VRAM Constraints

Problem:

Running LLM + Diffusion on a 4GB GPU exceeds memory limits.

Solution:

4-bit LLM quantization

Sequential model unloading

CPU offloading for diffusion

Attention slicing

FP16 compute

Memory Stability

Problem:

Diffusion models frequently crash on low VRAM GPUs.

Solution:

Sequential CPU offload

Reduced diffusion steps

Controlled resolution

Dynamic VRAM monitoring

Runtime Optimization

Problem:

Full pipelines can take 20+ minutes.

Solution:

SDXL Turbo

Low-step diffusion inference

Scene count optimization

Efficient FFmpeg stitching

Output Files

Each generation produces:

output/
 ├── images/
 │    ├── scene_1_image.png
 │    ├── scene_2_image.png
 │    └── scene_3_image.png
 │
 ├── audio/
 │    ├── audio_scene_1.mp3
 │    ├── audio_scene_2.mp3
 │    └── audio_scene_3.mp3
 │
 └── final_video_with_audio.mp4

Additional metadata:

scene_1_prompt.json
scene_2_prompt.json
scene_3_prompt.json
Future Improvements

Possible extensions:

Higher resolution generation

5-scene runtime mode

Faster diffusion schedulers

Better voice models

Multi-style video templates

GPU batch optimization

License

This project is for educational and research purposes.