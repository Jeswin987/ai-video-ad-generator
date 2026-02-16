"""
Model Manager with Singleton Pattern - optimized for GPU and 4-bit quantization.
Automatically detects CUDA and handles low-memory GPUs gracefully.
FIXED: Added use_gpu property for compatibility
"""

import logging
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional
from src.core.config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton Model Manager - loads model ONCE and reuses it.
    Uses local files only (no online download).
    """
    _instance = None
    _model = None
    _tokenizer = None
    _initialized = False
    _device = None  # Store device info

    def __new__(cls):
        """Singleton pattern - only one instance allowed"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model manager (only once)"""
        if not ModelManager._initialized:
            logger.info("🔧 Initializing ModelManager...")
            self._load_model()
            ModelManager._initialized = True

    def _load_model(self):
        """
        Load model and tokenizer from LOCAL files only.
        Supports 4-bit quantization and full precision fallback.
        """
        max_retries = Config.MAX_RETRIES
        retry_delay = Config.RETRY_DELAY

        # Auto-select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        Config.DEVICE = device
        ModelManager._device = device  # Store in class variable
        logger.info(f"🖥 Using device: {device}")

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"📥 Loading model (Attempt {attempt}/{max_retries})...")
                logger.info(f"📂 Model path: {Config.HF_MODEL_PATH}")

                # Verify path exists
                if not Config.HF_MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"Model directory not found: {Config.HF_MODEL_PATH}\n"
                        f"Please run: python download_gemma_2b.py"
                    )

                # Configure quantization
                quantization_config = None
                if Config.USE_4BIT_QUANTIZATION and device == "cuda":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("⚙️ Using 4-bit quantization for VRAM efficiency")
                else:
                    logger.info("⚙️ Using full precision (float16 on GPU, float32 on CPU)")

                # Load tokenizer
                logger.info("📖 Loading tokenizer from local files...")
                ModelManager._tokenizer = AutoTokenizer.from_pretrained(
                    str(Config.HF_MODEL_PATH),
                    local_files_only=True,
                    trust_remote_code=True
                )
                logger.info("✅ Tokenizer loaded successfully")

                # Load model
                logger.info("🤖 Loading model from local files...")
                ModelManager._model = AutoModelForCausalLM.from_pretrained(
                    str(Config.HF_MODEL_PATH),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="cuda" if device == "cuda" else None,
                    quantization_config=quantization_config,
                    local_files_only=True,
                    trust_remote_code=True
                )
                logger.info("✅ Model loaded successfully")

                # Log VRAM
                if device == "cuda":
                    allocated = torch.cuda.memory_allocated() / 1024 ** 3
                    reserved = torch.cuda.memory_reserved() / 1024 ** 3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                    logger.info(
                        f"💾 VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, total {total:.2f} GB"
                    )

                # Health check
                if not self._health_check():
                    raise RuntimeError("Model health check failed")

                logger.info("✅ ModelManager initialized successfully")
                return

            except FileNotFoundError as e:
                logger.error(f"❌ {str(e)}")
                logger.error("💡 Solution: Run 'python download_gemma_2b.py'")
                raise

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error("❌ CUDA out of memory!")
                    torch.cuda.empty_cache()
                    if attempt < max_retries:
                        logger.info(f"⏳ Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error("💡 Try reducing NUM_SCENES or enabling 4-bit mode")
                        raise
                else:
                    raise

            except Exception as e:
                logger.error(f"❌ Model loading error: {str(e)}")
                if attempt < max_retries:
                    logger.info(f"⏳ Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("❌ All retry attempts failed")
                    logger.error("💡 Troubleshooting:")
                    logger.error(f"  1. Check model path: {Config.HF_MODEL_PATH}")
                    logger.error("  2. Check GPU memory availability")
                    raise

    def _health_check(self) -> bool:
        """Run a small inference test"""
        try:
            logger.info("🏥 Running model health check...")
            test_input = "Hello"
            inputs = ModelManager._tokenizer(test_input, return_tensors="pt")
            
            if Config.DEVICE == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = ModelManager._model.generate(**inputs, max_new_tokens=5)
            
            result = ModelManager._tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ Health check passed: '{result[:40]}...'")
            return True
        
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

    @property
    def model(self):
        if ModelManager._model is None:
            raise RuntimeError("Model not loaded.")
        return ModelManager._model

    @property
    def tokenizer(self):
        if ModelManager._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        return ModelManager._tokenizer

    @property
    def use_gpu(self) -> bool:
        """Return whether GPU is being used - FOR COMPATIBILITY"""
        return ModelManager._device == "cuda" if ModelManager._device else torch.cuda.is_available()

    @property
    def device(self) -> str:
        """Return current device (cuda or cpu)"""
        return ModelManager._device if ModelManager._device else Config.DEVICE

    def get_memory_stats(self) -> dict:
        """Return GPU memory stats"""
        if not torch.cuda.is_available():
            return {"device": "cpu", "memory": "N/A"}
        
        try:
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            
            return {
                "device": "cuda",
                "allocated_gb": f"{allocated:.2f}",
                "reserved_gb": f"{reserved:.2f}",
                "total_gb": f"{total:.2f}",
                "free_gb": f"{total - reserved:.2f}"
            }
        except Exception as e:
            return {"device": "cuda", "error": str(e)}

    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU cache cleared")

    def reload_model(self):
        """Reload model manually"""
        logger.info("🔄 Reloading model...")
        ModelManager._model = None
        ModelManager._tokenizer = None
        ModelManager._initialized = False
        ModelManager._device = None
        self._load_model()