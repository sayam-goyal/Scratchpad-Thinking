import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from dataclasses import dataclass, field
from typing import Optional

# Set the device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="gpt2")
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=32)
    lora_init: bool = field(default=True)
    full_precision: bool = field(default=True)
    train: bool = field(default=False)
    ckpt_dir: str = field(default=".")
    token: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    batch_size: int = field(default=1)

@dataclass
class TrainingArguments:
    bf16: bool = field(default=True)
    model_max_length: int = field(default=512)
    num_latent: int = field(default=6)
    use_prj: bool = field(default=True)
    prj_dim: int = field(default=768)
    prj_no_ln: bool = field(default=False)
    prj_dropout: float = field(default=0.0)
    inf_latent_iterations: int = field(default=6)
    remove_eos: bool = field(default=True)
    use_lora: bool = field(default=True)
    greedy: bool = field(default=True)

def get_embd(model, model_name):
    """
    Retrieves the word embedding layer from the model.
    This is needed to convert token IDs back to embeddings during generation.
    """
    if "gpt2" in model_name:
        try:
            # For PeftModel
            return model.get_base_model().transformer.wte
        except AttributeError:
            # For regular model
            return model.transformer.wte
    else:
        # Fallback for other models, though this script is tailored for GPT-2
        try:
            return model.get_base_model().model.embed_tokens
        except AttributeError:
            return model.model.embed_tokens

class CODI(torch.nn.Module):
    """
    The main model class that wraps the base language model.
    """
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        # Load the base language model (e.g., gpt2)
        self.codi = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else torch.float32),
        )

        ori_vocab_size = self.codi.config.vocab_size

        # Define special tokens for the model's latent space
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1  # Beginning of Turn
        self.eot_id = ori_vocab_size + 2  # End of Turn

        # Resize token embeddings to include the new special tokens
        self.codi.resize_token_embeddings(ori_vocab_size + 3)
        self.dim = self.codi.config.hidden_size

        # Apply LoRA to the model for efficient fine-tuning adaptation
        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        # Optional projection layer for latent embeddings
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.training_args.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))

    def get_embd(self, model, model_name):
        """Helper function to get the embedding layer."""
        return get_embd(model, model_name)

def run_inference():
    
    print("Initializing arguments...")
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingArguments()

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", 'c_fc'],
        init_lora_weights=True,
    )

    print("Initializing the CODI model...")
    model = CODI(model_args, training_args, lora_config)

    print("Loading model checkpoint...")
    ckpt_path = os.path.expanduser(model_args.ckpt_dir)
    try:
        # Try loading SafeTensors first
        state_dict_path = os.path.join(ckpt_path, "model.safetensors")
        state_dict = load_file(state_dict_path)
    except FileNotFoundError:
        # Fallback to PyTorch binary
        state_dict_path = os.path.join(ckpt_path, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location=device)
    except Exception as e:
        print(f"Error: Could not find or load model checkpoint at {ckpt_path}.")
        print(f"Please ensure the path is correct and the files 'model.safetensors' or 'pytorch_model.bin' exist.")
        print(f"Original error: {e}")
        return

    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    print("Checkpoint loaded successfully.")

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id

    model.to(device)
    model.to(torch.bfloat16)
    model.eval()
    print("Model is ready for inference.")

    try:
        question = input("\nPlease enter your question: ")
    except EOFError:
        print("\nNo input received. Exiting.")
        return
        
    print("\nTokenizing input...")
    inputs = tokenizer(question, return_tensors="pt", padding="longest")

    bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(inputs["input_ids"].size(0), 1)
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Generating response...")
    with torch.no_grad():
        #Encode the question into an initial latent embedding
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        for _ in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        eot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.eot_id], dtype=torch.long, device=device)
        ).unsqueeze(0).expand(inputs["input_ids"].size(0), -1, -1)
        
        output_emb = eot_emb
        
        pred_tokens = []
        max_new_tokens = 256
        for i in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            # Get logits for the last token in the sequence
            logits = out.logits[:, -1, :]

            next_token_id = torch.argmax(logits, dim=-1).squeeze()
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            pred_tokens.append(next_token_id.item())

            output_emb = model.get_embd(model.codi, model.model_name)(next_token_id).unsqueeze(0).unsqueeze(0)

    response = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    print("\n--- Model Response ---")
    print(response)
    print("----------------------\n")


if __name__ == "__main__":
    run_inference()
