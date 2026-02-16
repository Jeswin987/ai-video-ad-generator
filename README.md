AI Video Marketing Ad Generator

Creating professional marketing videos requires scripting, visual design, voiceover generation, and editing. This project automates the entire pipeline using fully local generative AI models, transforming a single text prompt into a complete, ready-to-use marketing video.

Key Characteristics

Fully local inference (no external APIs)

RTX 2050 (4GB VRAM) compatible

5-scene structured storytelling for a 30-second ad

Gemma 2 2B IT (FP16) for scene and script generation

SDXL Turbo (512x512) for fast diffusion-based image synthesis

Peak VRAM usage: ~3.5–4GB

Average generation time: ~10 minutes per 15–40s video


System Architecture:

User Prompt
   ↓
Flask REST API
   ↓
VideoGenerationPipeline (Core Orchestrator)
   ↓
Storyline & Scene Structuring (Gemma 2B IT – FP16)
   ↓
Scene Normalization (5 Structured Scenes)
   ↓
Image Generation per Scene (SDXL Turbo @ 512x512)
   ↓
Voiceover Synthesis & Duration Alignment
   ↓
Scene-Level Video Composition (Motion + Overlay)
   ↓
Media Stitching (FFmpeg)
   ↓
Final Rendered Marketing Video (MP4)
