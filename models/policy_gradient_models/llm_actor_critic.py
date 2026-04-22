import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

class LLMActorCriticNetwork(nn.Module):
    """
    LLM-based Actor-Critic network.
    Wraps a frozen Vision-Language Model (Gemma-4) as a backbone,
    extracts the final hidden state, and passes it through trainable
    Actor and Critic heads.
    """
    def __init__(self, model_id="google/gemma-4-E2B", load_in_4bit=True, freeze_backbone=True):
        super().__init__()
        
        print(f"[LLMActorCritic] Loading processor for {model_id}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            print("[LLMActorCritic] Found processor locally.")
        except Exception:
            print("[LLMActorCritic] Processor not found locally. Downloading from Hugging Face...")
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
        
        print(f"[LLMActorCritic] Loading backbone model {model_id}...")
        
        # Determine model loading kwargs
        model_kwargs = {
            "device_map": "cuda:0",
            "dtype": torch.bfloat16,
        }
        
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
        try:
            self.backbone = AutoModelForImageTextToText.from_pretrained(
                model_id,
                local_files_only=True,
                **model_kwargs
            )
            print("[LLMActorCritic] Found model locally.")
        except Exception:
            print("[LLMActorCritic] Model not found locally. Downloading from Hugging Face...")
            self.backbone = AutoModelForImageTextToText.from_pretrained(
                model_id,
                local_files_only=False,
                **model_kwargs
            )
            
        if freeze_backbone:
            print("[LLMActorCritic] Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Get hidden size from config
        try:
            hidden_size = self.backbone.config.text_config.hidden_size
        except AttributeError:
            hidden_size = self.backbone.config.hidden_size
            
        print(f"[LLMActorCritic] Backbone hidden size: {hidden_size}")
            
        # --- Heads ---
        # Actor head predicts 5 actions (North, South, East, West, Stop)
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        # Critic head predicts 1 value
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Ensure heads are on the same device as backbone
        self.actor_fc.to(self.backbone.device)
        self.critic_fc.to(self.backbone.device)

    def forward(self, return_both=True, **kwargs):
        """
        Forward pass.
        Expects batched inputs prepared by the processor.
        """
        # Build a clean dict with all tensors moved to the right device
        device = self.backbone.device
        model_inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

        with torch.set_grad_enabled(self.backbone.training and any(p.requires_grad for p in self.backbone.parameters())):
            outputs = self.backbone(
                output_hidden_states=True,
                **model_inputs
            )
            
            # Extract last hidden state
            # shape: (batch_size, seq_len, hidden_size)
            last_hidden_state = outputs.hidden_states[-1]
            
            # Take the representation of the last token in the sequence
            context_vector = last_hidden_state[:, -1, :]
            
        # Cast to float32 - backbone is bfloat16 but trainable heads are float32
        context_vector = context_vector.float()

        # Forward through trainable heads
        actor_out = torch.relu(self.actor_fc[0](context_vector))
        actor_logits = self.actor_fc[2](actor_out)
        
        probs = torch.softmax(actor_logits, dim=-1)
        log_probs = torch.log_softmax(actor_logits, dim=-1)
        
        self.last_log_probs = log_probs # Store for PPO update
        
        if return_both:
            critic_out = torch.relu(self.critic_fc[0](context_vector))
            values = self.critic_fc[2](critic_out)
            return probs, values
            
        return probs

    def state_dict(self, *args, **kwargs):
        """Override to only save the heads, preventing massive checkpoints."""
        state_dict = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}
        
    def load_state_dict(self, state_dict, strict=False):
        """Override to load without complaining about missing backbone weights."""
        return super().load_state_dict(state_dict, strict=strict)
