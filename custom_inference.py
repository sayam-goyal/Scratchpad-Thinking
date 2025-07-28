import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from dataclasses import dataclass, field
from typing import Optional
import copy

# Set the device to CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Class Definitions (from model.py) ---
# These classes are needed to load the model structure correctly.

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="gpt2")
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=32)
    lora_init: bool = field(default=True)
    full_precision: bool = field(default=True)
    train: bool = field(default=False)
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

# --- Main Inference Logic ---

def run_inference():
    """
    Main function to set up the model and run inference with detailed probing.
    """
    # =========================================================================
    # CHOOSE YOUR MODEL CHECKPOINT HERE
    # =========================================================================
    ckpt_dir = "."
    # =========================================================================

    # 1. Initialize arguments
    print("Initializing arguments...")
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingArguments()

    # 2. Configure LoRA
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

    # 3. Initialize the CODI model
    print("Initializing the CODI model...")
    model = CODI(model_args, training_args, lora_config)

    # 4. Load the fine-tuned model weights (checkpoint)
    print(f"Loading model checkpoint from: {ckpt_dir}...")
    ckpt_path = os.path.expanduser(ckpt_dir)
    try:
        state_dict_path = os.path.join(ckpt_path, "model.safetensors")
        state_dict = load_file(state_dict_path)
    except FileNotFoundError:
        state_dict_path = os.path.join(ckpt_path, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location=device)
    except Exception as e:
        print(f"Error: Could not find or load model checkpoint at {ckpt_path}.")
        print(f"Original error: {e}")
        return

    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    print("Checkpoint loaded successfully.")

    # 5. Initialize the tokenizer
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

    # 6. Move model to device and set to eval mode
    model.to(device)
    model.to(torch.bfloat16)
    model.eval()
    print("\nModel is ready for inference.")
    
    # MODIFICATION: Start a loop for continuous question asking
    while True:
        # 7. Get user input
        try:
            question = input("\nPlease enter your question (or 'q' to quit): ")
            if question.lower().strip() == 'q':
                print("Exiting.")
                break
        except EOFError:
            print("\nNo input received. Exiting.")
            break
            
        # 8. Prepare the input data
        print("\nTokenizing input...")
        inputs = tokenizer(question, return_tensors="pt", padding="longest")

        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(inputs["input_ids"].size(0), 1)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 9. Run modified inference with probing
        print("Starting modified inference process...")
        with torch.no_grad():
            # A. Initial Forward Pass to process the prompt
            outputs = model.codi(
                input_ids=inputs["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                attention_mask=inputs["attention_mask"]
            )

            # B. Get the initial state for the main loop
            main_past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # C. Helper function for printing top k tokens
            def probe_and_print_top_k(logits, tokenizer, k=5):
                probabilities = F.softmax(logits.to(torch.float32), dim=-1)
                top_k_probs, top_k_ids = torch.topk(probabilities, k)
                top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids.squeeze())
                print("    Top 5 candidates:")
                for i in range(k):
                    token_str = top_k_tokens[i].replace('Ġ', ' ')
                    print(f"      - '{token_str}' (Probability: {(top_k_probs.squeeze()[i] * 100):.2f}%)")

            # D. Analyze the "0th" latent thought (initial state)
            print(f"\n--- Analyzing Initial State (Latent Thought 0) ---")
            print("  [Probe 1/2] Projecting initial state to vocabulary space...")
            initial_logits = outputs.logits[:, -1, :]
            probe_and_print_top_k(initial_logits, tokenizer)

            print("  [Probe 2/2] Starting temporary autoregressive generation from this state...")
            probe_past_key_values_0 = copy.deepcopy(main_past_key_values)
            probe_output_emb_0 = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.eot_id], dtype=torch.long, device=device)
            ).unsqueeze(0).expand(inputs["input_ids"].size(0), -1, -1)

            max_probe_tokens = 32
            temp_generated_tokens_0 = []
            final_probe_outputs_0 = None
            second_to_last_probe_outputs_0 = None

            for _ in range(max_probe_tokens):
                second_to_last_probe_outputs_0 = final_probe_outputs_0
                probe_outputs_0 = model.codi(
                    inputs_embeds=probe_output_emb_0,
                    use_cache=True,
                    past_key_values=probe_past_key_values_0
                )
                final_probe_outputs_0 = probe_outputs_0
                probe_past_key_values_0 = probe_outputs_0.past_key_values
                next_token_id_0 = torch.argmax(probe_outputs_0.logits[:, -1, :], dim=-1).squeeze()
                if next_token_id_0.item() == tokenizer.eos_token_id:
                    break
                temp_generated_tokens_0.append(next_token_id_0.item())
                probe_output_emb_0 = model.get_embd(model.codi, model.model_name)(next_token_id_0).unsqueeze(0).unsqueeze(0)

            temp_response_0 = tokenizer.decode(temp_generated_tokens_0, skip_special_tokens=True)
            print(f"    Temporarily generated answer: '{temp_response_0}'")
            last_word_0 = temp_response_0.split()[-1] if temp_response_0 else "N/A"
            print(f"    Top 5 candidates for the final token ('{last_word_0}'):")
            logits_to_probe_0 = None
            if second_to_last_probe_outputs_0 is not None:
                logits_to_probe_0 = second_to_last_probe_outputs_0.logits[:, -1, :]
            elif final_probe_outputs_0 is not None:
                logits_to_probe_0 = final_probe_outputs_0.logits[:, -1, :]
            if logits_to_probe_0 is not None:
                probe_and_print_top_k(logits_to_probe_0, tokenizer)
            else:
                print("    No temporary output was generated.")


            # E. Main Probing Loop for thoughts 1 through 6
            for i in range(training_args.inf_latent_iterations):
                print(f"\n--- Analyzing Latent Thought {i + 1}/{training_args.inf_latent_iterations} ---")

                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=main_past_key_values
                )
                main_past_key_values = outputs.past_key_values

                print("  [Probe 1/2] Projecting latent thought to vocabulary space...")
                latent_logits = outputs.logits[:, -1, :]
                probe_and_print_top_k(latent_logits, tokenizer)

                print("  [Probe 2/2] Starting temporary autoregressive generation from this state...")
                
                probe_past_key_values = copy.deepcopy(main_past_key_values)
                probe_output_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device=device)
                ).unsqueeze(0).expand(inputs["input_ids"].size(0), -1, -1)

                temp_generated_tokens = []
                final_probe_outputs = None
                second_to_last_probe_outputs = None

                for _ in range(max_probe_tokens):
                    second_to_last_probe_outputs = final_probe_outputs
                    probe_outputs = model.codi(
                        inputs_embeds=probe_output_emb,
                        use_cache=True,
                        past_key_values=probe_past_key_values
                    )
                    final_probe_outputs = probe_outputs
                    probe_past_key_values = probe_outputs.past_key_values
                    next_token_id = torch.argmax(probe_outputs.logits[:, -1, :], dim=-1).squeeze()
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
                    temp_generated_tokens.append(next_token_id.item())
                    probe_output_emb = model.get_embd(model.codi, model.model_name)(next_token_id).unsqueeze(0).unsqueeze(0)

                temp_response = tokenizer.decode(temp_generated_tokens, skip_special_tokens=True)
                print(f"    Temporarily generated answer: '{temp_response}'")
                last_word = temp_response.split()[-1] if temp_response else "N/A"
                print(f"    Top 5 candidates for the final token ('{last_word}'):")

                logits_to_probe = None
                if second_to_last_probe_outputs is not None:
                    logits_to_probe = second_to_last_probe_outputs.logits[:, -1, :]
                elif final_probe_outputs is not None:
                    logits_to_probe = final_probe_outputs.logits[:, -1, :]
                
                if logits_to_probe is not None:
                    probe_and_print_top_k(logits_to_probe, tokenizer)
                else:
                    print("    No temporary output was generated.")

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # F. Final Answer Generation (resumes normal process)
            print("\n--- Latent thought process complete. Generating final response... ---")
            
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
                    past_key_values=main_past_key_values
                )
                main_past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1).squeeze()
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                pred_tokens.append(next_token_id.item())
                output_emb = model.get_embd(model.codi, model.model_name)(next_token_id).unsqueeze(0).unsqueeze(0)

        # 10. Decode and print the final response
        response = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        print("\n--- Model Response ---")
        print(response)
        print("----------------------\n")


if __name__ == "__main__":
    run_inference()
