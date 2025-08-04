#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset
from accelerate.utils import set_seed
from safetensors.torch import load_file


import numpy as np
#from scipy.stats import mode

from Patching.src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    PatchingArguments,
)

do_print = True
probe_topk = 5
probe_idx_original = None
probe_idx_modified = None
test_attention = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def evaluation(model_args, data_args, training_args, patching_args, current_patch_idx: int = -1):
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError
    
    model = CODI(model_args, training_args, lora_config)
    #if "llama" in model_args.model_name_or_path:
    #    model.codi.resize_token_embeddings(128261)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    
    tokenizer_path = model_args.model_name_or_path 
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"

        # Load original dataset
    if data_args.data_name.endswith(".json"):
        with open(data_args.data_name, "r") as f:
            data_original = json.load(f)
        test_set_original = [
        {"question": q, "cot": c, "answer": a}
        for q, c, a in zip(data_original["question"], data_original["cot"], data_original["answer"])] 
    elif "zen-E/GSM8k-Aug" in data_args.data_name:
        dataset_original = load_dataset(data_args.data_name)
        test_set_original = dataset_original['test']
    else:
        raise NotImplementedError
    
    # Load modified dataset (if patching is enabled)
    test_set_modified = None
    if current_patch_idx != -1 and patching_args.data_name_modified:
        if patching_args.data_name_modified.endswith(".json"):
            with open(patching_args.data_name_modified, "r") as f:
                data_modified = json.load(f)
            test_set_modified = [
            {"question": q, "cot": c, "answer": a}
            for q, c, a in zip(data_modified["question"], data_modified["cot"], data_modified["answer"])] 
        elif "zen-E/GSM8k-Aug" in patching_args.data_name_modified:
            dataset_modified = load_dataset(patching_args.data_name_modified)
            test_set_modified = dataset_modified['test']
        else:
            raise NotImplementedError
        
        if len(test_set_original) != len(test_set_modified):
            raise ValueError("Original and modified datasets must have the same number of examples for patching.")



    logging.warning("Formatting inputs...")
    question_original = [] 
    answer_original = []
    procedures_original = []

    question_modified = []
    answer_modified = []
    procedures_modified = []

    # get numerical answer for original
    for example in test_set_original:
        question_original.append(f"{example[question_name].strip().replace('  ', ' ')}")
        answer_original.append(float(example[answer_name].replace(",", "")))
        procedures_original.append(example["cot"])
        
    # get numerical answer for modified
    if test_set_modified: #segun yo esto es por si no quiero hacer la comparativa y que siga el transcurso normal
        for example in test_set_modified:
            question_modified.append(f"{example[question_name].strip().replace('  ', ' ')}")
            answer_modified.append(float(example[answer_name].replace(",", "")))
            procedures_modified.append(example["cot"])

        
    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question_original)/data_args.batch_size)
    logging.warning(f"Total example: {len(question_original)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    
    question_data_original = []
    question_data_modified = []

    for i in range(eval_step):
        #Esto hace falta cachar a lo bien
        if i < eval_step - 1:
            batch_original = tokenizer(
                question_original[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch_original = tokenizer(
                question_original[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_original["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch_original["input_ids"].size(0), 2)
        batch_original["input_ids"] = torch.cat((batch_original["input_ids"], bot_tensor), dim=1)
        batch_original["attention_mask"] = torch.cat((batch_original["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch_original['input_len'] = len(batch_original['input_ids'][0])
        question_data_original.append(batch_original.to(device))

        if test_set_modified:
            if i < eval_step - 1:
                batch_modified = tokenizer(
                    question_modified[i*data_args.batch_size: (i+1)*data_args.batch_size],
                    return_tensors="pt",
                    padding="longest",
                )
            else:
                batch_modified = tokenizer(
                    question_modified[i*data_args.batch_size:],
                    return_tensors="pt",
                    padding="longest",
                )
            
            if training_args.remove_eos:
                bot_tensor_mod = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_modified["input_ids"].size(0), 1)
            else:
                bot_tensor_mod = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch_modified["input_ids"].size(0), 2)
            batch_modified["input_ids"] = torch.cat((batch_modified["input_ids"], bot_tensor_mod), dim=1)
            batch_modified["attention_mask"] = torch.cat((batch_modified["attention_mask"], torch.ones_like(bot_tensor_mod)), dim=1)
            batch_modified['input_len'] = len(batch_modified['input_ids'][0])
            question_data_modified.append(batch_modified.to(device))


    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list_original = []
    ans_pred_list_modified = []
    
    len_cot_original = []
    len_cot_modified = []

    log_original = []
    log_modified = []
    
    log_count_original = 0
    log_count_modified = 0
    #model.eval()
    #top5_indices_list_decoded = []
    top5_indices_list_decoded_original = []
    top5_indices_list_decoded_modified = []

    for step, batch_original in enumerate(question_data_original):
        batch_size = batch_original["input_ids"].size(0)
        #top5_values_list, top5_indices_list = [], []
        top5_values_list_original, top5_indices_list_original = [], []
        top5_values_list_modified, top5_indices_list_modified = [], []
        with torch.no_grad():
            # Initialize for original path
            past_key_values_original = None
            latent_embd_original = None
            outputs_original = model.codi(input_ids=batch_original["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values_original, attention_mask=batch_original["attention_mask"])
            past_key_values_original = outputs_original.past_key_values
            latent_embd_original = outputs_original.hidden_states[-1][:, -1, :].unsqueeze(1)
            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd_original), dim=-1)  
            top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
            top5_values_list_original.append(top5_values)
            top5_indices_list_original.append(top5_indices)
            if training_args.use_prj:
                latent_embd_original = model.prj(latent_embd_original)

            # Initialize for modified path (if patching)
            past_key_values_modified = None
            latent_embd_modified = None
            if current_patch_idx != -1:
                batch_modified = question_data_modified[step] # Get corresponding modified batch
                outputs_modified = model.codi(input_ids=batch_modified["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values_modified, attention_mask=batch_modified["attention_mask"])
                past_key_values_modified = outputs_modified.past_key_values
                latent_embd_modified = outputs_modified.hidden_states[-1][:, -1, :].unsqueeze(1)

                probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd_modified), dim=-1)  
                top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
                top5_values_list_modified.append(top5_values)
                top5_indices_list_modified.append(top5_indices)

                if training_args.use_prj:
                    latent_embd_modified = model.prj(latent_embd_modified)
            
            # Iterate the latent thoughts
            inf_latent_iterations = training_args.inf_latent_iterations #NOS QUEDAMOS ACA PORQUE NO ENTIENDO 
            for i in range(inf_latent_iterations):
                # Apply patching
                if i == current_patch_idx and current_patch_idx != -1:
                    # Swap latent_embd and past_key_values
                    temp_latent_embd = latent_embd_original
                    #temp_past_key_values = past_key_values_original

                    latent_embd_original = latent_embd_modified #Ahora lo unico que me qudda duda aca. Es saber si se esta sobrescribiendo. Y si, porque lo que se pasa es un latente que tiene contextualizacio pasada. Es decir, haber hecho una resta ya asi
                    #past_key_values_original = past_key_values_modified

                    latent_embd_modified = temp_latent_embd
                    #past_key_values_modified = temp_past_key_values
                    
                    if do_print:
                        print(f"--- PATCHING AT LATENT ITERATION {i} ---")

                 # Process for original path
                outputs_original = model.codi(inputs_embeds=latent_embd_original, use_cache=True, output_hidden_states=True, past_key_values=past_key_values_original)
                past_key_values_original = outputs_original.past_key_values
                latent_embd_original = outputs_original.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                # Probe the latent thought before the projection
                probs_original = torch.nn.functional.softmax(model.codi.lm_head(latent_embd_original), dim=-1)
                top5_values, top5_indices = torch.topk(probs_original, k=probe_topk, dim=2)
                top5_values_list_original.append(top5_values)
                top5_indices_list_original.append(top5_indices)

                if training_args.use_prj:
                    latent_embd_original = model.prj(latent_embd_original)

                # Process for modified path (if patching)
                if current_patch_idx != -1:
                    outputs_modified = model.codi(inputs_embeds=latent_embd_modified, use_cache=True, output_hidden_states=True, past_key_values=past_key_values_modified)
                    past_key_values_modified = outputs_modified.past_key_values
                    latent_embd_modified = outputs_modified.hidden_states[-1][:, -1, :].unsqueeze(1)

                    probs_modified = torch.nn.functional.softmax(model.codi.lm_head(latent_embd_modified), dim=-1)
                    top5_values, top5_indices = torch.topk(probs_modified, k=probe_topk, dim=2)
                    top5_values_list_modified.append(top5_values)
                    top5_indices_list_modified.append(top5_indices)
                    if training_args.use_prj:
                        latent_embd_modified = model.prj(latent_embd_modified)

                # Probe the latent thought before the projection
                #probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
                #top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
                #top5_values_list.append(top5_values)
                #top5_indices_list.append(top5_indices)

                #if training_args.use_prj:
                    #latent_embd = model.prj(latent_embd)

            # --- Decode for Original Path ---
            if training_args.remove_eos:
                eot_emb_original = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            else:
                eot_emb_original = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            eot_emb_original = eot_emb_original.expand(batch_size, -1, -1)

            output_original = eot_emb_original
            
            seq_len_original = 0
            finished_original = torch.zeros(batch_size, dtype=torch.bool, device="cuda")  # Track EOS for each sequence
            pred_tokens_original = [[] for _ in range(batch_size)]

            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len_original += 1

                out_original = model.codi(
                        inputs_embeds=output_original,
                        output_hidden_states=False,
                        attention_mask=None,
                        use_cache=True,
                        output_attentions=False,
                        past_key_values=past_key_values_original
                    )
                past_key_values_original = out_original.past_key_values
                logits_original = out_original.logits[:, -1, :model.codi.config.vocab_size-1]

                if training_args.greedy:
                    next_token_ids_original = torch.argmax(logits_original, dim=-1).squeeze(-1)
                else:
                    logits_original /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values_original, _ = torch.topk(logits_original, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value_original = top_k_values_original[:, -1].unsqueeze(-1)
                        logits_original[logits_original < min_top_k_value_original] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit_original, sorted_indices_original = torch.sort(logits_original, descending=True, dim=-1)
                        cumulative_probs_original = torch.cumsum(F.softmax(sorted_logit_original, dim=-1), dim=-1)

                        sorted_indices_to_remove_original = cumulative_probs_original > gen_kwargs["top_p"]
                        if sorted_indices_to_remove_original.any():
                            sorted_indices_to_remove_original = sorted_indices_to_remove_original.roll(1, dims=-1)
                            sorted_indices_to_remove_original[:, 0] = False

                        for b in range(batch_size):
                            logits_original[b, sorted_indices_original[b, sorted_indices_to_remove_original[b]]] = -float("inf")
                    
                    probs_original = F.softmax(logits_original, dim=-1)
                    next_token_ids_original = torch.multinomial(probs_original, num_samples=1).squeeze(-1)

                for b in range(batch_size):
                    if not finished_original[b]:
                        pred_tokens_original[b].append(next_token_ids_original[b].item())
                        if next_token_ids_original[b] == tokenizer.eos_token_id:
                            finished_original[b] = True

                if finished_original.all():
                    break
                output_original = model.get_embd(model.codi, model.model_name)(next_token_ids_original).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens_original):
                len_cot_original.append(len(pred_token))
                decoded_pred_original = tokenizer.decode(pred_token, skip_special_tokens=True)
                if do_print:
                    print(f"--- Original Path - Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question_original[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred_original)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred_original)}; Groundtruth={answer_original[step*data_args.batch_size+mini_step]}")
                    print("")
                ans_pred_list_original.append(extract_answer_number(decoded_pred_original))
                
            top5_values_list_original = torch.cat(top5_values_list_original, dim=1)
            top5_indices_list_original = torch.cat(top5_indices_list_original, dim=1)

            if probe_idx_original is not None:
                top5_values_list_original = top5_values_list_original[:, probe_idx_original]
                top5_indices_list_original = top5_indices_list_original[:, probe_idx_original]
                top5_values_list_original = top5_values_list_original.unsqueeze(1)
                top5_indices_list_original = top5_indices_list_original.unsqueeze(1)

            # decode top5_indices_list
            for ii in range(len(top5_indices_list_original)): # batch
                do_log=True
                #if int(answer_original[log_count_original]) != int(extract_answer_number(tokenizer.decode(pred_tokens_original[ii]))):
                    #do_log=False
                if do_log:
                    log_original.append(f"Question{log_count_original}...")
                    log_original.append(f"{question_original[log_count_original]}...")
                    log_original.append(f"CoT={procedures_original[log_count_original]}, Answer={answer_original[log_count_original]}")
                log_count_original += 1
                top5_indices_list_decoded_tmp_original = []
                for jj in range(top5_indices_list_original.size(1)):
                    if do_log:
                        if test_attention:
                            log_original.append(f"decoded {jj}th latent's attended tokens (top5): {attn_to_lats[jj][ii]}")
                        log_original.append(f"decoded {jj}th latent (top5): {[tokenizer.decode(x) for x in top5_indices_list_original[ii, jj]]}")
                    for kk in range(top5_indices_list_original.size(2)):
                        top5_indices_list_decoded_tmp_original.append(tokenizer.decode(top5_indices_list_original[ii, jj, kk]))
                top5_indices_list_decoded_original.append(top5_indices_list_decoded_tmp_original)
                if do_log:
                    if test_attention:
                        log_original.append(f"decoded before answer token's attended tokens (top5): {attn_to_lats[-1][ii]}")
                    log_original.append(f"Model Prediction: {tokenizer.decode(pred_tokens_original[ii])}")
                    log_original.append("\n\n")

            # --- Decode for Modified Path (if patching) ---
            if current_patch_idx != -1:
                if training_args.remove_eos:
                    eot_emb_modified = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
                else:
                    eot_emb_modified = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
                eot_emb_modified = eot_emb_modified.expand(batch_size, -1, -1)

                output_modified = eot_emb_modified
                
                seq_len_modified = 0
                finished_modified = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
                pred_tokens_modified = [[] for _ in range(batch_size)]

                for i in range(gen_kwargs["max_new_tokens"]):
                    seq_len_modified += 1

                    out_modified = model.codi(
                            inputs_embeds=output_modified,
                            output_hidden_states=False,
                            attention_mask=None,
                            use_cache=True,
                            output_attentions=False,
                            past_key_values=past_key_values_modified
                        )
                    past_key_values_modified = out_modified.past_key_values
                    logits_modified = out_modified.logits[:, -1, :model.codi.config.vocab_size-1]

                    if training_args.greedy:
                        next_token_ids_modified = torch.argmax(logits_modified, dim=-1).squeeze(-1)
                    else:
                        logits_modified /= gen_kwargs["temperature"]
                        if gen_kwargs["top_k"] > 1:
                            top_k_values_modified, _ = torch.topk(logits_modified, gen_kwargs["top_k"], dim=-1)
                            min_top_k_value_modified = top_k_values_modified[:, -1].unsqueeze(-1)
                            logits_modified[logits_modified < min_top_k_value_modified] = -float("inf")

                        if gen_kwargs["top_p"] < 1.0:
                            sorted_logit_modified, sorted_indices_modified = torch.sort(logits_modified, descending=True, dim=-1)
                            cumulative_probs_modified = torch.cumsum(F.softmax(sorted_logit_modified, dim=-1), dim=-1)

                            sorted_indices_to_remove_modified = cumulative_probs_modified > gen_kwargs["top_p"]
                            if sorted_indices_to_remove_modified.any():
                                sorted_indices_to_remove_modified = sorted_indices_to_remove_modified.roll(1, dims=-1)
                                sorted_indices_to_remove_modified[:, 0] = False

                            for b in range(batch_size):
                                logits_modified[b, sorted_indices_modified[b, sorted_indices_to_remove_modified[b]]] = -float("inf")
                        
                        probs_modified = F.softmax(logits_modified, dim=-1)
                        next_token_ids_modified = torch.multinomial(probs_modified, num_samples=1).squeeze(-1)

                    for b in range(batch_size):
                        if not finished_modified[b]:
                            pred_tokens_modified[b].append(next_token_ids_modified[b].item())
                            if next_token_ids_modified[b] == tokenizer.eos_token_id:
                                finished_modified[b] = True

                    if finished_modified.all():
                        break
                    output_modified = model.get_embd(model.codi, model.model_name)(next_token_ids_modified).unsqueeze(1).to(device)

                for mini_step, pred_token in enumerate(pred_tokens_modified):
                    len_cot_modified.append(len(pred_token))
                    decoded_pred_modified = tokenizer.decode(pred_token, skip_special_tokens=True)
                    if do_print:
                        print(f"--- Modified Path - Question {step*data_args.batch_size+mini_step} Starts...")
                        print(f"Q: {question_modified[step*data_args.batch_size+mini_step]}")
                        print(decoded_pred_modified)
                        print(f"Question {step*data_args.batch_size+mini_step} Ends")
                        print(f"Prediction={extract_answer_number(decoded_pred_modified)}; Groundtruth={answer_modified[step*data_args.batch_size+mini_step]}")
                        print("")
                    ans_pred_list_modified.append(extract_answer_number(decoded_pred_modified))

            top5_values_list_modified = torch.cat(top5_values_list_modified, dim=1)
            top5_indices_list_modified = torch.cat(top5_indices_list_modified, dim=1)

            if probe_idx_modified is not None:
                top5_values_list_modified = top5_values_list_modified[:, probe_idx_modified]
                top5_indices_list_modified = top5_indices_list_modified[:, probe_idx_modified]
                top5_values_list_modified = top5_values_list_modified.unsqueeze(1)
                top5_indices_list_modified = top5_indices_list_modified.unsqueeze(1)

            # decode top5_indices_list
            for ii in range(len(top5_indices_list_modified)): # batch
                do_log=True
                #if int(answer_modified[log_count_modified]) != int(extract_answer_number(tokenizer.decode(pred_tokens_modified[ii]))):
                    #do_log=False
                if do_log:
                    log_modified.append(f"Question{log_count_modified}...")
                    log_modified.append(f"{question_modified[log_count_modified]}...")
                    log_modified.append(f"CoT={procedures_modified[log_count_modified]}, Answer={answer_modified[log_count_modified]}")
                log_count_modified += 1
                top5_indices_list_decoded_tmp_modified = []
                for jj in range(top5_indices_list_modified.size(1)):
                    if do_log:
                        if test_attention:
                            log_modified.append(f"decoded {jj}th latent's attended tokens (top5): {attn_to_lats[jj][ii]}")
                        log_modified.append(f"decoded {jj}th latent (top5): {[tokenizer.decode(x) for x in top5_indices_list_modified[ii, jj]]}")
                    for kk in range(top5_indices_list_modified.size(2)):
                        top5_indices_list_decoded_tmp_modified.append(tokenizer.decode(top5_indices_list_modified[ii, jj, kk]))
                top5_indices_list_decoded_modified.append(top5_indices_list_decoded_tmp_modified)
                if do_log:
                    if test_attention:
                        log_modified.append(f"decoded before answer token's attended tokens (top5): {attn_to_lats[-1][ii]}")
                    log_modified.append(f"Model Prediction: {tokenizer.decode(pred_tokens_modified[ii])}")
                    log_modified.append("\n\n")

    accuracy_original = compute_accuracy(answer_original, ans_pred_list_original)
    print(f"adapter: {model_args.adapter_name_or_path} | Original GSM8K test accuracy: {100*accuracy_original:.2f}% | ")
    print(f"average length of COT (Original): {sum(len_cot_original)/len(len_cot_original)}")

    with open(f"Patching/decoded_latent_original_patch_{current_patch_idx}.txt", "w") as f:
        f.write("\n".join(log_original))

    if current_patch_idx!= -1:
        accuracy_modified = compute_accuracy(answer_modified, ans_pred_list_modified)
        print(f"adapter: {model_args.adapter_name_or_path} | Modified GSM8K test accuracy (after patching): {100*accuracy_modified:.2f}% | ")
        print(f"average length of COT (Modified after patching): {sum(len_cot_modified)/len(len_cot_modified)}")
        with open(f"Patching/decoded_latent_modified_patch_{current_patch_idx}.txt", "w") as f:
            f.write("\n".join(log_modified))

    return 100*accuracy_original, (100*accuracy_modified if current_patch_idx != -1 else None)

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    # use the last number as the answer
    pred_answer = float(pred[-1])

    return pred_answer


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, PatchingArguments))
    model_args, data_args, training_args, patching_args = parser.parse_args_into_dataclasses()

    # The original check for patch_latent_idx and data_name_modified needs adjustment.
    # We will now loop through patch_latent_idx, so data_name_modified is only required if we are patching at all.
    if patching_args.start_patch_idx != patching_args.end_patch_idx and not patching_args.data_name_modified:
        raise ValueError("If iterative patching is enabled (start_patch_idx != end_patch_idx), data_name_modified must also be provided.")
    if patching_args.end_patch_idx > training_args.inf_latent_iterations:
        raise ValueError(f"end_patch_idx ({patching_args.end_patch_idx}) must be less than or equal to inf_latent_iterations ({training_args.inf_latent_iterations}).")

    all_accuracies_original = []
    all_accuracies_modified = []
    patch_indices = []

    # Loop through the latent indices for patching
    for current_idx in range(patching_args.start_patch_idx, patching_args.end_patch_idx+1):
        print(f"\n--- Running evaluation with patching at latent index: {current_idx} ---")
        accu_list_original_iter = []
        accu_list_modified_iter = []
        for i in range(training_args.inf_num_iterations):
            # Pass the current_idx to the evaluation function
            accu_original, accu_modified = evaluation(model_args, data_args, training_args, patching_args, current_idx)
            accu_list_original_iter.append(accu_original)
            if accu_modified is not None:
                accu_list_modified_iter.append(accu_modified)
        
        avg_acc_original = sum(accu_list_original_iter) / len(accu_list_original_iter)
        print(f"Average accuracy over {training_args.inf_num_iterations} sampling (Original) for patch_idx {current_idx}: {avg_acc_original:.2f}%")
        all_accuracies_original.append(avg_acc_original)
        
        if accu_list_modified_iter: # Only append if actual patching happened
            avg_acc_modified = sum(accu_list_modified_iter) / len(accu_list_modified_iter)
            print(f"Average accuracy over {training_args.inf_num_iterations} sampling (Modified after patching) for patch_idx {current_idx}: {avg_acc_modified:.2f}%")
            all_accuracies_modified.append(avg_acc_modified)
        
        patch_indices.append(current_idx)

    print("\n--- Summary of Results ---")
    for i, idx in enumerate(patch_indices):
        print(f"Patch Index {idx}: Original Accuracy = {all_accuracies_original[i]:.2f}%")
        if all_accuracies_modified:
            print(f"Patch Index {idx}: Modified Accuracy = {all_accuracies_modified[i]:.2f}%")

