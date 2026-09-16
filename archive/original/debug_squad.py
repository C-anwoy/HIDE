import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import _settings
import dataeval.SQuAD as SQuAD

def run_diagnostics():
    model_name = 'llama3-8b'
    device = 'cuda:0'
    
    model_dict = {'llama3-8b': "/home/models/Meta-Llama-3-8B"}
    model_path = os.path.join(_settings.MODEL_PATH, model_dict[model_name])
    
    print(f"1. Loading tokenizer and model in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="eager" 
    ).to(device)
    model.eval()

    print(f"\n2. Loading 1 SQuAD sample...")
    dataset = SQuAD.get_dataset(tokenizer).select(range(1))
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))

    input_ids = batch['input_ids'].to(device)
    attn_mask = batch['attention_mask'].to(device)
    I = input_ids.shape[1]
    
    print(f"   Input Length (I): {I}")
    print(f"   Input Mask sum: {attn_mask.sum().item()} (Should be > 0)")
    print(f"   Has NaNs in input? {torch.isnan(input_ids.float()).any().item()}")

    gen_config_dict = SQuAD._generate_config(tokenizer)
    gen_config_dict['max_new_tokens'] = 32
    gen_config_dict['pad_token_id'] = tokenizer.eos_token_id
    gen_config = GenerationConfig(**gen_config_dict)

    print(f"\n3. Running model.generate()...")
    with torch.no_grad():
        dict_outputs = model.generate(
            input_ids, 
            attention_mask=attn_mask,
            num_beams=1, do_sample=False, 
            generation_config=gen_config,
            return_dict_in_generate=True
        )
        
        O = dict_outputs.sequences.shape[1] - I
        print(f"   Generated Tokens (O): {O}")
        if O == 0:
            print("   [CRITICAL] Model generated 0 tokens!")
            return
        
        full_ids = dict_outputs.sequences.to(device)
        gen_mask = torch.ones((1, O), device=device, dtype=attn_mask.dtype)
        full_attn_mask = torch.cat([attn_mask, gen_mask], dim=1)
        
        print(f"\n4. Running second forward pass...")
        forward_out = model(full_ids, attention_mask=full_attn_mask, output_attentions=True, output_hidden_states=True)
        
        l_mid = model.config.num_hidden_layers // 2
        attn_matrix = forward_out.attentions[l_mid] 
        
        print(f"\n5. Analyzing Attention Matrix (Layer {l_mid})...")
        print(f"   Shape: {attn_matrix.shape}")
        
        has_nans = torch.isnan(attn_matrix).any().item()
        print(f"   Has NaNs? {has_nans}")
        if not has_nans:
            print(f"   Min: {attn_matrix.min().item():.4f}, Max: {attn_matrix.max().item():.4f}")
        
        cross_attn = attn_matrix[:, I:, :I] 
        print(f"\n   Cross-Attn Shape: {cross_attn.shape}")
        
        cross_has_nans = torch.isnan(cross_attn).any().item()
        print(f"   Cross-Attn Has NaNs? {cross_has_nans}")
        if not cross_has_nans:
            print(f"   Cross-Attn Min: {cross_attn.min().item():.4f}, Max: {cross_attn.max().item():.4f}")
        
        avg_cross_attn = cross_attn.mean(dim=0) 
        Omega = avg_cross_attn.sum(dim=1).mean().item()
        print(f"   Calculated Omega: {Omega}")

        H_prev = forward_out.hidden_states[l_mid - 1] 
        print(f"\n6. Analyzing Hidden States (Layer {l_mid-1})...")
        print(f"   Has NaNs? {torch.isnan(H_prev).any().item()}")
        
        H_input = H_prev[:I, :] 
        flow_updates = torch.matmul(avg_cross_attn, H_input.to(torch.bfloat16)) 
        print(f"   Flow Updates Has NaNs? {torch.isnan(flow_updates).any().item()}")
        
        Delta_in = torch.norm(flow_updates, p=2, dim=1).mean().item()
        print(f"   Calculated Delta_in: {Delta_in}")

if __name__ == '__main__':
    run_diagnostics()