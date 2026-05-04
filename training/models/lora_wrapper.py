#!/usr/bin/env python3
"""
LoRA Wrapper for HiDream-E1 Finetuning

Loads HiDream-I1 base model and configures LoRA adaptors for efficient finetuning.
"""

# CRITICAL: Disable flash-attn BEFORE any imports
import os
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["DIFFUSERS_ATTN_IMPLEMENTATION"] = "eager"

import torch
import sys
from pathlib import Path
from typing import Dict, Optional
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from diffusers import AutoencoderKL, HiDreamImageTransformer2DModel, UniPCMultistepScheduler

# Add HiDream-E1 path for pipeline
E1_PATH = Path(__file__).parent.parent.parent / "HiDream-E1"
if E1_PATH.exists():
    sys.path.insert(0, str(E1_PATH))

try:
    from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
except ImportError:
    print("⚠️  Warning: Could not import HiDreamImageEditingPipeline")
    print("   Make sure HiDream-E1 pipeline is available")
    HiDreamImageEditingPipeline = None


class HiDreamE1LoRA:
    """HiDream-E1 model wrapper with LoRA for finetuning"""
    
    def __init__(self, config, device="cuda"):
        """Initialize model with LoRA
        
        Args:
            config: Training configuration object
            device: Device to load model on
        """
        self.config = config
        
        # For DDP training, use LOCAL_RANK to assign specific GPU
        import os
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            self.device = torch.device(f'cuda:{local_rank}')
            print(f"\n📍 DDP Mode: Process assigned to GPU {local_rank}")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.dtype = getattr(torch, config.model.dtype)
        
        print(f"\n{'='*60}")
        print(f"LOADING HiDream-E1 WITH LORA")
        print(f"{'='*60}\n")
        
        # Load components
        print("📝 Loading tokenizers and text encoders...")
        self.tokenizers = self._load_tokenizers()
        self.text_encoders = self._load_text_encoders()
        
        print("🎨 Loading VAE...")
        self.vae = self._load_vae()
        
        print("🔧 Loading transformer with LoRA...")
        self.transformer = self._load_transformer_with_lora()
        
        print("⏱️  Loading scheduler...")
        self.scheduler = self._load_scheduler()
        
        print("🔗 Creating pipeline...")
        self.pipeline = self._create_pipeline()
        
        # Move to device
        self._move_to_device()
        
        # DEBUG: Verify pipeline components after device move
        if self.pipeline is not None:
            print("\n🔍 DEBUG: Verifying pipeline components...")
            print(f"  - Pipeline type: {type(self.pipeline).__name__}")
            print(f"  - text_encoder exists: {hasattr(self.pipeline, 'text_encoder') and self.pipeline.text_encoder is not None}")
            print(f"  - text_encoder_2 exists: {hasattr(self.pipeline, 'text_encoder_2') and self.pipeline.text_encoder_2 is not None}")
            print(f"  - text_encoder_3 exists: {hasattr(self.pipeline, 'text_encoder_3') and self.pipeline.text_encoder_3 is not None}")
            print(f"  - text_encoder_4 exists: {hasattr(self.pipeline, 'text_encoder_4') and self.pipeline.text_encoder_4 is not None}")
            if hasattr(self.pipeline, 'text_encoder') and self.pipeline.text_encoder is not None:
                print(f"  - text_encoder device: {self.pipeline.text_encoder.device}")
            
            # Test encode_prompt immediately
            print("\n🧪 Testing encode_prompt with sample input...")
            try:
                test_result = self.pipeline.encode_prompt(
                    prompt=["test sunset scene"],
                    prompt_2=["test sunset scene"],
                    prompt_3=["test sunset scene"],
                    prompt_4=["test sunset scene"],
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                print(f"  ✅ encode_prompt test successful!")
                print(f"  - Returns {len(test_result)} values")
                for i, val in enumerate(test_result):
                    if val is not None:
                        print(f"  - Output {i}: shape {val.shape}, dtype {val.dtype}")
                    else:
                        print(f"  - Output {i}: None ⚠️")
            except Exception as e:
                print(f"  ❌ encode_prompt test FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        # Print parameter info
        self._print_parameter_info()
        
        print(f"\n{'='*60}")
        print(f"✅ MODEL LOADED SUCCESSFULLY")
        print(f"{'='*60}\n")
    
    def _load_tokenizers(self) -> Dict:
        """Load all tokenizers"""
        tokenizers = {}
        
        # CLIP tokenizers (1 & 2)
        from transformers import CLIPTokenizer
        tokenizers['tokenizer'] = CLIPTokenizer.from_pretrained(
            self.config.model.base_model_path,
            subfolder="tokenizer",
        )
        tokenizers['tokenizer_2'] = CLIPTokenizer.from_pretrained(
            self.config.model.base_model_path,
            subfolder="tokenizer_2",
        )
        
        # T5 tokenizer (3)
        from transformers import T5Tokenizer
        tokenizers['tokenizer_3'] = T5Tokenizer.from_pretrained(
            self.config.model.base_model_path,
            subfolder="tokenizer_3",
        )
        
        # Llama tokenizer (4) - for instructions
        tokenizers['tokenizer_4'] = PreTrainedTokenizerFast.from_pretrained(
            self.config.model.llama_model
        )
        
        return tokenizers
    
    def _load_text_encoders(self) -> Dict:
        """Load all text encoders"""
        text_encoders = {}
        
        # CLIP text encoders
        from transformers import CLIPTextModelWithProjection
        text_encoders['text_encoder'] = CLIPTextModelWithProjection.from_pretrained(
            self.config.model.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        )
        text_encoders['text_encoder_2'] = CLIPTextModelWithProjection.from_pretrained(
            self.config.model.base_model_path,
            subfolder="text_encoder_2",
            torch_dtype=self.dtype,
        )
        
        # T5 encoder
        from transformers import T5EncoderModel
        text_encoders['text_encoder_3'] = T5EncoderModel.from_pretrained(
            self.config.model.base_model_path,
            subfolder="text_encoder_3",
            torch_dtype=self.dtype,
        )
        
        # Llama encoder (for instruction understanding)
        text_encoders['text_encoder_4'] = LlamaForCausalLM.from_pretrained(
            self.config.model.llama_model,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=self.dtype,
        )
        
        # Freeze all text encoders (we only train transformer LoRA)
        for encoder in text_encoders.values():
            encoder.requires_grad_(False)
            encoder.eval()
        
        return text_encoders
    
    def _load_vae(self) -> AutoencoderKL:
        """Load VAE (frozen)"""
        vae = AutoencoderKL.from_pretrained(
            self.config.model.base_model_path,
            subfolder="vae",
            torch_dtype=self.dtype,
        )
        
        # Freeze VAE
        vae.requires_grad_(False)
        vae.eval()
        
        return vae
    
    def _load_transformer_with_lora(self) -> HiDreamImageTransformer2DModel:
        """Load transformer and add LoRA adaptors"""
        # Load base transformer
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            self.config.model.base_model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        
        # Configure LoRA (no task_type needed for diffusion models)
        # Convert OmegaConf types to plain Python to avoid JSON serialization issues
        lora_config = LoraConfig(
            r=int(self.config.lora.rank),
            lora_alpha=int(self.config.lora.alpha),
            target_modules=list(self.config.lora.target_modules),  # Convert ListConfig to list
            lora_dropout=float(self.config.lora.dropout),
            bias=str(self.config.lora.bias),
        )
        
        # Add LoRA to transformer
        try:
            transformer = get_peft_model(transformer, lora_config)
            print(f"  ✅ Added LoRA adaptors (rank={self.config.lora.rank})")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not add LoRA with PEFT: {e}")
            print(f"  Using manual LoRA configuration")
            # Fallback: Add LoRA manually using transformer.add_adapter()
            transformer.add_adapter(lora_config)
        
        # Enable gradient checkpointing if configured
        if self.config.training.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
            print(f"  ✅ Enabled gradient checkpointing")
        
        return transformer
    
    def _load_scheduler(self):
        """Load diffusion scheduler"""
        scheduler = UniPCMultistepScheduler.from_pretrained(
            self.config.model.base_model_path,
            subfolder="scheduler",
        )
        return scheduler
    
    def _create_pipeline(self):
        """Create HiDreamImageEditingPipeline"""
        if HiDreamImageEditingPipeline is None:
            print("  ⚠️  Warning: HiDreamImageEditingPipeline not available")
            return None
        
        pipeline = HiDreamImageEditingPipeline(
            vae=self.vae,
            text_encoder=self.text_encoders['text_encoder'],
            text_encoder_2=self.text_encoders['text_encoder_2'],
            text_encoder_3=self.text_encoders['text_encoder_3'],
            text_encoder_4=self.text_encoders['text_encoder_4'],
            tokenizer=self.tokenizers['tokenizer'],
            tokenizer_2=self.tokenizers['tokenizer_2'],
            tokenizer_3=self.tokenizers['tokenizer_3'],
            tokenizer_4=self.tokenizers['tokenizer_4'],
            transformer=self.transformer,
            scheduler=self.scheduler,
        )
        
        return pipeline
    
    def _move_to_device(self):
        """Move models to device (device already set correctly in __init__ for DDP)"""
        self.vae.to(self.device)
        self.transformer.to(self.device)
        
        for encoder in self.text_encoders.values():
            encoder.to(self.device)
    
    def _print_parameter_info(self):
        """Print trainable parameter information"""
        total_params = sum(p.numel() for p in self.transformer.parameters())
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        
        print(f"\n📊 Parameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def get_trainable_parameters(self):
        """Return iterator over trainable parameters"""
        return filter(lambda p: p.requires_grad, self.transformer.parameters())
    
    def save_lora_weights(self, output_path: str):
        """Save only LoRA weights"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        if hasattr(self.transformer, 'save_pretrained'):
            self.transformer.save_pretrained(output_path)
            print(f"✅ Saved LoRA weights to {output_path}")
        else:
            # Fallback: save state dict
            torch.save(
                {k: v for k, v in self.transformer.state_dict().items() if 'lora' in k.lower()},
                output_path / "lora_weights.pt"
            )
            print(f"✅ Saved LoRA weights to {output_path}/lora_weights.pt")
    
    def load_lora_weights(self, lora_path: str):
        """Load LoRA weights"""
        lora_path = Path(lora_path)
        
        if (lora_path / "adapter_config.json").exists():
            # Load PEFT adapter
            from peft import PeftModel
            self.transformer = PeftModel.from_pretrained(self.transformer, lora_path)
            print(f"✅ Loaded LoRA weights from {lora_path}")
        elif (lora_path / "lora_weights.pt").exists():
            # Load manual state dict
            state_dict = torch.load(lora_path / "lora_weights.pt")
            self.transformer.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded LoRA weights from {lora_path}/lora_weights.pt")
        else:
            raise ValueError(f"No LoRA weights found at {lora_path}")


def load_model_with_lora(config, device="cuda"):
    """Convenience function to load model with LoRA
    
    Args:
        config: Training configuration
        device: Device to load on
        
    Returns:
        HiDreamE1LoRA instance
    """
    return HiDreamE1LoRA(config, device=device)


if __name__ == "__main__":
    # Test model loading
    print("Testing HiDreamE1LoRA...")
    
    import sys
    sys.path.append("../..")
    from omegaconf import OmegaConf
    
    config = OmegaConf.load("../config/training_config.yaml")
    
    model = HiDreamE1LoRA(config)
    
    print("\n✅ Model loading test complete!")

