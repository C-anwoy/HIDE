import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import glob
import json
import os
import copy
import time

import pandas as pd
import torch
import tqdm
import transformers
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore

import _settings
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import dataeval.race as race
import dataeval.halueval as halueval
import models
import utils
from func.metric import *
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama3-8b')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)
parser.add_argument('--continue_from', type=str, default=None, 
                    help='Path to a _partial.pkl file to continue from after an error')
parser.add_argument('--keywords', type=int, default=20)
parser.add_argument('--layer', type=int, default=16)
parser.add_argument('--kernel', type=str, default='rbf')



args = parser.parse_args()
if args.continue_from == None:
    mode = "w+"
else:
    mode = "a+"

# Ensure the directory exists
log_dir = os.path.join(_settings.GENERATION_FOLDER, "logs")
os.makedirs(log_dir, exist_ok=True)

# Define the log file path
log_path = os.path.join(log_dir, "logInfo_{}_{}_{}.txt".format(args.model, args.dataset, args.project_ind))

# Open the log file
logInfo = open(log_path, mode=mode, encoding="utf-8")

def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset
    if data_name == "SQuAD":
        return SQuAD.get_dataset
    if data_name == "race":
        return race.get_dataset
    if data_name == "halueval":
        return halueval.get_dataset


def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    if data_name == 'SQuAD':
        generation_config = SQuAD._generate_config(tokenizer)
    if data_name == 'race':
        generation_config = race._generate_config(tokenizer)
    if data_name == 'halueval':
        generation_config = halueval._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config


@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt, cache_dir=None, run_id=None):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    
    # 'sentence-transformers/nli-roberta-large'
    SenSimModel = SentenceTransformer(os.path.join(_settings.MODEL_PATH, 'nli-roberta-large'), device = device)
    bertscore = BERTScore(model_name_or_path=os.path.join(_settings.MODEL_PATH, 'bert-base-uncased'), device=device)

    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    time_start=time.time()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue

        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        generation_config = transformers.GenerationConfig(**generation_config)
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            greedy_start_time = time.time()
            dict_outputs = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                        num_beams=1,
                                        do_sample=False,
                                        generation_config=generation_config,
                                        output_hidden_states = True,
                                        return_dict_in_generate=True,
                                        output_scores=True)
            greedy_generation_time = time.time() - greedy_start_time

            scores = dict_outputs.scores

            perplexity_start_time = time.time()
            perplexity = get_perplexity_score(scores)
            perplexity_time = time.time() - perplexity_start_time

            energy_start_time = time.time()
            energy_score = get_energy_score(scores)
            energy_time = time.time() - energy_start_time

            hidden_states = dict_outputs.hidden_states
            most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]

            input_tokens = dict_outputs.sequences.cpu()[0, 0:input_length]
            
            hsic_start_time = time.time()
            hsic_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = get_unbiased_hsic_score_keybert(hidden_states, tokenizer, input_tokens, most_likely_generations, keywords = args.keywords, layer=args.layer, kernel=args.kernel)
            hsic_time = time.time() - hsic_start_time

        torch.cuda.empty_cache()
        generations = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            multiple_generation_start_time = time.time()
            dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                            num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                            temperature=args.temperature, generation_config=generation_config,
                            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                            )
            multiple_generation_time = time.time() - multiple_generation_start_time

            generation = dict_outputs.sequences[:, input_length:].cpu()
            generations.append(generation)
            num_tokens = get_num_tokens(generation)
            scores = dict_outputs.scores

            entropy_start_time = time.time()
            predictive_entropy = get_lenghthNormalized_entropy(scores, num_tokens) 
            entropy_time = time.time() - entropy_start_time

            hidden_states = dict_outputs.hidden_states

            eigen_start_time = time.time()
            eigenIndicator, eigenValue = getEigenIndicator_v0(hidden_states, num_tokens)
            eigen_time = time.time() - eigen_start_time

            num_gens -= len(generation)

        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
        generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        
        lexical_sim_start_time = time.time()
        lexical_similarity = getLexicalSim(generated_texts)
        lexical_sim_time = time.time() - lexical_sim_start_time

        bert_score_start_time = time.time()
        sent_bertscore = getAvgBertScore(bertscore, best_generated_text, generated_texts)
        bert_score_time = time.time() - bert_score_start_time
        
        eigen_output_start_time = time.time()
        eigenIndicatorOutput, eigenValue_O = getEigenIndicatorOutput(generated_texts, SenSimModel)
        eigen_output_time = time.time() - eigen_output_start_time

        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                generations=generated_texts,
            )
        )
        curr_seq.update(
            dict(
                perplexity=perplexity
            )
        )
        curr_seq.update(
            dict(
                energy=energy_score
            )
        )
        curr_seq.update(
            dict(
                hsic=hsic_score
            )
        )
        curr_seq.update(
            dict(
                input_keywords=input_keywords,
                output_keywords=output_keywords,
                input_tokens_topk=input_tokens_topk, 
                output_tokens_topk=output_tokens_topk
            )
        )
        curr_seq.update(
            dict(
                lexical_similarity=lexical_similarity
            )
        )
        curr_seq.update(
            dict(
                sent_bertscore=sent_bertscore
            )
        )
        curr_seq.update(
            dict(
                entropy=predictive_entropy
            )
        )
        curr_seq.update(
            dict(
                eigenIndicator=eigenIndicator
            )
        )
        curr_seq.update(
            dict(
                eigenIndicatorOutput=eigenIndicatorOutput
            )
        )
        # Add timing information
        curr_seq.update(
            dict(
                greedy_generation_time=greedy_generation_time,
                multiple_generation_time=multiple_generation_time,
                perplexity_time=perplexity_time,
                energy_time=energy_time,
                hsic_time=hsic_time,
                entropy_time=entropy_time,
                lexical_sim_time=lexical_sim_time,
                bert_score_time=bert_score_time,
                eigen_time=eigen_time,
                eigen_output_time=eigen_output_time
            )
        )
        if args.dataset == 'coqa' or args.dataset == "TruthfulQA":
            print("additional_answers:", batch['additional_answers'])
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]
        if args.dataset == 'umwp':
            curr_seq['answerable'] = [x for x in batch['answerable']]
        sequences.append(curr_seq)
        
        # After processing each batch (after adding to sequences)
        if batch_idx % 10 == 0 and batch_idx > 0 and cache_dir is not None and run_id is not None:
            partial_path = os.path.join(cache_dir, f'{run_id}_partial.pkl')
            pd.to_pickle(sequences, partial_path)
            print(f"Saved intermediate results after {batch_idx} batches to {partial_path}")

        torch.cuda.empty_cache()

        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=logInfo)
        print("Question:", batch['question'][0], file=logInfo)
        print("GTAns:", batch['answer'][0], file=logInfo)
        print("BestAns:", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True), file=logInfo)
        # print("BatchGenerations:", generated_texts, file=logInfo)
        print("Perplexity:", perplexity, file=logInfo)
        print("Energy:", energy_score, file=logInfo)
        print("HSIC:", hsic_score, file=logInfo)
        print("input keywords:", input_keywords, file=logInfo)
        print("output keywords:", output_keywords, file=logInfo)
        print("input tokens topk:", input_tokens_topk, file=logInfo)
        print("output tokens topk:", output_tokens_topk, file=logInfo)
        # print("input tokens topk duplication:", input_tokens_topk_duplication, file=logInfo)
        # print("output tokens topk duplication:", output_tokens_topk_duplication, file=logInfo)
        print("NormalizedEntropy: ", predictive_entropy, file=logInfo)
        print("LexicalSimilarity: ", lexical_similarity, file=logInfo)
        print("SentBERTScore: ", sent_bertscore, file=logInfo)
        print("EigenScore: ", eigenIndicator, file=logInfo)
        print("EigenValue:", eigenValue, file=logInfo)
        print("EigenScore-Output: ", eigenIndicatorOutput, file=logInfo)
        print("\n","\n","\n", file=logInfo)
    return sequences


def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens


def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}')
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences, cache_dir=cache_dir, run_id=run_id)
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess, continue_from=args.continue_from)
