import os
import datasets
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

def get_dataset(tokenizer):
    cache = os.path.join(settings.data_folder(), 'nq_open')
    if os.path.exists(cache):
        data = datasets.load_from_disk(cache)
    else:
        data = datasets.load_dataset('nq_open', split='validation')
        data.save_to_disk(cache)
    id_map = {_['question']: str(i) for (i, _) in enumerate(data)}

    def process_instance(example):
        example['id'] = id_map[example['question']]
        all_answers = example.pop('answer')
        example['additional_answers'] = all_answers[1:]
        example['answer'] = all_answers[0]
        example['prompt'] = sample_to_prompt({k: example[k] for k in ['question']})
        inputs = tokenizer(example['prompt'], padding=False, truncation=False)
        outputs = tokenizer(all_answers[0], padding=False, truncation=False)
        example['input_ids'] = inputs['input_ids']
        example['attention_mask'] = inputs.attention_mask
        example['labels'] = outputs.input_ids.copy()
        example['labels'] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example['labels']]
        return example
    data = data.map(process_instance, load_from_cache_file=False)
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
    return data
