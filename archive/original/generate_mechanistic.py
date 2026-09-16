import os
import time
import argparse
import pandas as pd
import torch
import tqdm
from transformers import GenerationConfig

import _settings
import dataeval.nq_open as nq_open
import dataeval.SQuAD as SQuAD
import models
# Use the exact function from your metric.py to ensure kwargs match
from func.metric import * 
from sentence_transformers import SentenceTransformer, util

def load_model_eager(model_name, device):
    """Forces eager attention to ensure output_attentions=True works."""
    from _settings import MODEL_PATH
    model_dict = {'llama3-8b': "/home/models/Meta-Llama-3-8B", 'gemma-2-9b': "/home/models/gemma-2-9b"}
    model_path = os.path.join(MODEL_PATH, model_dict.get(model_name, model_name))
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name} with EAGER attention in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, # <-- CRITICAL FIX: bfloat16 prevents NaN overflow on long SQuAD contexts
        attn_implementation="eager" 
    )
    model.to(device)
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3-8b')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--layer', type=int, default=16)
    args = parser.parse_args()

    out_dir = os.path.join(_settings.GENERATION_FOLDER, "mechanistic")
    os.makedirs(out_dir, exist_ok=True)
    
    model, tokenizer = load_model_eager(args.model, args.device)
    sensim_model = SentenceTransformer(os.path.join(_settings.MODEL_PATH, 'nli-roberta-large'), device=args.device)

    l_mid = model.config.num_hidden_layers // 2 if hasattr(model.config, 'num_hidden_layers') else args.layer
    print(f"Using layer {l_mid} for mechanistic metrics.")

    datasets_to_run = {'nq_open': nq_open} #'SQuAD': SQuAD, 

    for ds_name, ds_module in datasets_to_run.items():
        print(f"\n--- Starting Fast Mechanistic Extraction for {ds_name} ---")
        dataset = ds_module.get_dataset(tokenizer)
        dataset = dataset.shuffle(seed=42).select(range(min(args.samples, len(dataset))))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        records = []
        save_path = os.path.join(out_dir, f"{args.model}_{ds_name}_mechanistic.csv")

        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch['input_ids'].to(args.device)
            I = input_ids.shape[1]

            gen_config_dict = ds_module._generate_config(tokenizer)
            gen_config_dict['max_new_tokens'] = 64
            gen_config_dict['pad_token_id'] = tokenizer.eos_token_id
            gen_config = GenerationConfig(**gen_config_dict)

            with torch.no_grad():
                dict_outputs = model.generate(
                    input_ids, 
                    attention_mask=batch['attention_mask'].to(args.device),
                    num_beams=1, do_sample=False, 
                    generation_config=gen_config,
                    output_hidden_states=True, return_dict_in_generate=True
                )
                
                generated_ids = dict_outputs.sequences[0, I:]
                O = len(generated_ids)
                
                if O == 0: 
                    continue 

                input_tokens = dict_outputs.sequences[0, :I]
                hidden_states = dict_outputs.hidden_states
                
                try:
                    hsic_score, _, _, _, _ = get_unbiased_hsic_score_keybert(
                        hidden_states, tokenizer, input_tokens, generated_ids, 
                        keywords=20, layer=args.layer, kernel='rbf'
                    )
                except Exception as e:
                    print(f"HSIC Error on batch {batch_idx}: {e}")
                    hsic_score = 0.0 

               # Mechanistic Metrics Forward Pass
                full_ids = dict_outputs.sequences.to(args.device)
                
                # Correct Mask
                input_mask = batch['attention_mask'].to(args.device)
                gen_mask = torch.ones((1, O), device=args.device, dtype=input_mask.dtype)
                full_attn_mask = torch.cat([input_mask, gen_mask], dim=1)
                
                forward_out = model(full_ids, attention_mask=full_attn_mask, output_attentions=True, output_hidden_states=True)
                
                # --- BULLETPROOF MATRIX EXTRACTION ---
                attn_matrix = forward_out.attentions[l_mid]
                if isinstance(attn_matrix, tuple):
                    attn_matrix = attn_matrix
                
                # Force slice: Batch 0, All Heads, Output Queries (I:), Input Keys (:I)
                cross_attn = attn_matrix[0, :, I:, :I] 
                avg_cross_attn = cross_attn.mean(dim=0) # [O, I]
                
                Omega = avg_cross_attn.sum(dim=1).mean().item()

                H_prev = forward_out.hidden_states[l_mid - 1]
                if isinstance(H_prev, tuple):
                    H_prev = H_prev
                    
                # Force slice: Batch 0, Input Sequence (:I), All Hidden Dims
                H_input = H_prev[0, :I, :] # [I, hidden_dim]
                
                flow_updates = torch.matmul(avg_cross_attn, H_input.to(torch.bfloat16)) 
                Delta_in = torch.norm(flow_updates, p=2, dim=1).mean().item()

                # Ground Truth
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                ans_gt = batch['answer'] 
                
                gen_emb = sensim_model.encode(generated_text, convert_to_tensor=True)
                ans_emb = sensim_model.encode(ans_gt, convert_to_tensor=True)
                sim = util.cos_sim(gen_emb, ans_emb).item()
                is_correct = 1 if sim > 0.9 else 0

                # Fallback to catch any remaining NaNs
                if torch.isnan(torch.tensor(Omega)):
                    Omega = 0.0
                if torch.isnan(torch.tensor(Delta_in)):
                    Delta_in = 0.0

                records.append({
                    "id": batch['id'],
                    "question": batch['question'],
                    "answer": ans_gt,
                    "generated_text": generated_text,
                    "is_correct": is_correct,
                    "HIDE_score": hsic_score,
                    "Omega": Omega,
                    "Delta_in": Delta_in
                })

            if batch_idx > 0 and batch_idx % 20 == 0:
                pd.DataFrame(records).to_csv(save_path, index=False)
            torch.cuda.empty_cache()

        pd.DataFrame(records).to_csv(save_path, index=False)
        print(f"Saved {len(records)} records to {save_path}")

if __name__ == "__main__":
    main()