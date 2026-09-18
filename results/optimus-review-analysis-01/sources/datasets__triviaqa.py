import os
import datasets
import pandas as pd
from hide import settings

def sample_to_prompt(sample, **kwargs):
    if isinstance(sample['question'], list):
        return [sample_to_prompt({'question': _}, **kwargs) for _ in sample['question']]
    return f"Answer these questions:\nQ: {sample['question']}\nA:"

def _generate_config(tokenizer):
    eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', '.']]
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']]
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)

def process_data_to_model_inputs(batch, tokenizer):
    answers = [answer['value'] for answer in batch['answer']]
    batch['additional_answers'] = [answer.get('aliases', []) for answer in batch['answer']]
    batch_with_prompt = sample_to_prompt(batch)
    inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
    outputs = tokenizer(answers, padding=False, truncation=False)
    batch['input_ids'] = inputs.input_ids
    batch['attention_mask'] = inputs.attention_mask
    batch['decoder_input_ids'] = outputs.input_ids
    batch['decoder_attention_mask'] = outputs.attention_mask
    batch['labels'] = outputs.input_ids.copy()
    batch['answer'] = answers
    batch['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch['labels']]
    batch['id'] = batch['question_id']
    batch['prompt'] = batch_with_prompt
    return batch

def get_dataset(tokenizer, split='validation'):
    cache = os.path.join(settings.data_folder(), 'trivia_qa')
    if os.path.exists(cache):
        data = datasets.load_from_disk(cache)
    else:
        data = datasets.load_dataset('trivia_qa', 'rc.nocontext', split=split)
        data.save_to_disk(cache)
    id_mem = set()

    def remove_dups(batch):
        if batch['question_id'][0] in id_mem:
            return {_: [] for _ in batch.keys()}
        id_mem.add(batch['question_id'][0])
        return batch
    data = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    assert pd.Series([_['question_id'] for _ in data]).value_counts().max() == 1
    data = data.map(lambda _: process_data_to_model_inputs(_, tokenizer), batched=True, batch_size=10, load_from_cache_file=False, remove_columns=['search_results', 'question_source', 'entity_pages'])
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels'], output_all_columns=True)
    return data
