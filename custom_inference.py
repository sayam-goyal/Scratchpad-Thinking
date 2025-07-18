import torch
import transformers
from safetensors.torch import load_file
from src.model import CODI, ModelArguments, TrainingArguments

def extract_answer_number(sentence: str):
    import re
    # Try to extract a number (float/int) from the sentence
    match = re.search(r"[-+]?[.]?\d+([.,]\d+)?", sentence)
    if match:
        return float(match.group().replace(",", ""))
    return sentence.strip()

# Step 1: Set up model and tokenizer arguments
model_args = ModelArguments(
    model_name_or_path="gpt2",  # Change to your model
    lora_init=True,
    lora_r=128,
    lora_alpha=32,
    token=None,
    ckpt_dir="."  # Change to your checkpoint
)
training_args = TrainingArguments(
    model_max_length=512,
    remove_eos=True,
    use_prj=True,
    prj_dim=768,
    prj_no_ln=False,
    prj_dropout=0.0,
    inf_latent_iterations=6,
    greedy=True,
)

# Step 2: Load model and tokenizer
lora_config = None
if model_args.lora_init:
    lora_config = transformers.LoraConfig(
        task_type=transformers.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )
model = CODI(model_args, training_args, lora_config)
try:
    state_dict = load_file(f"{model_args.ckpt_dir}/model.safetensors")
except Exception:
    state_dict = torch.load(f"{model_args.ckpt_dir}/pytorch_model.bin")
model.load_state_dict(state_dict, strict=False)
model.codi.tie_weights()

# Step 3: Load tokenizer

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    token=model_args.token,
    model_max_length=training_args.model_max_length,
    padding_side="left",
    use_fast=False,
)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = model.pad_id if hasattr(model, 'pad_id') else tokenizer.convert_tokens_to_ids('[PAD]')

# Step 4: Prepare your custom question
custom_question = "What is 17 plus 25?"

# Step 5: Tokenize the question
batch = tokenizer([custom_question], return_tensors="pt", padding="longest")
if training_args.remove_eos:
    bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
else:
    bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
batch = batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model = model.to(batch["input_ids"].device)
model.eval()

# Step 6: Run inference
with torch.no_grad():
    past_key_values = None
    outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
    past_key_values = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    if training_args.use_prj:
        latent_embd = model.prj(latent_embd)
    for i in range(training_args.inf_latent_iterations):
        outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)
    # Prepare for generation
    eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device=latent_embd.device)).unsqueeze(0)
    eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)
    output = eot_emb
    pred_tokens = []
    finished = torch.zeros(1, dtype=torch.bool, device=output.device)
    for _ in range(256):
        out = model.codi(inputs_embeds=output, use_cache=True, past_key_values=past_key_values)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
        next_token_id = torch.argmax(logits, dim=-1).squeeze(-1)
        pred_tokens.append(next_token_id.item())
        if next_token_id == tokenizer.eos_token_id:
            break
        output = model.get_embd(model.codi, model.model_name)(next_token_id).unsqueeze(1)
    decoded_pred = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    print(f"Q: {custom_question}")
    print(f"Model answer: {decoded_pred}")
    print(f"Extracted answer: {extract_answer_number(decoded_pred)}")
