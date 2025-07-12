import functools
import json
import os
import ast

import datasets
import pandas as pd
from datasets import Dataset

import _settings

def _save_dataset():
    save_path = f'{_settings.DATA_FOLDER}/RACE'
    if not os.path.exists(save_path):
        data = datasets.load_dataset("EleutherAI/race", split='test')
        

        dataset = {}

        dataset['story'] = []
        dataset['question'] = []
        dataset['answer'] = []
        # dataset['additional_answers'] = []
        dataset['id'] = []
        dataset['options'] = []

        
        for sample_id, sample in enumerate(data):
            story = sample['article']
            problems_str = sample['problems']
            # Convert string to Python list using ast.literal_eval
            questions = ast.literal_eval(problems_str)
            # answers = sample['answers']
            # additional_answers = sample['additional_answers']
            for question_index, question in enumerate(questions):
                dataset['story'].append(story)
                dataset['question'].append(question['question'])
                options = question['options']
                dataset['options'].append(options)
                answer_idx = question['answer']
                dataset['answer'].append(options[ord(answer_idx) - ord('A')])
                dataset['id'].append(str(sample_id) + '_' + str(question_index))
                
        dataset_df = pd.DataFrame.from_dict(dataset)

        dataset = Dataset.from_pandas(dataset_df)

        dataset.save_to_disk(save_path)
    return save_path

@functools.lru_cache(1)
def read_all_contexts():
    dataset = datasets.load_from_disk(_save_dataset())
    return {_['id']: _['story'] for _ in dataset}

def sample_to_prompt(sample, **kwargs):

    prompt = f"{sample['story']} Q: {sample['question']} \nOptions: {', '.join(sample['options'])} \nA:"
    return prompt

def get_dataset(tokenizer):
    dataset = datasets.load_from_disk(_save_dataset())
    # id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

    def encode_race(example):
        example['prompt'] = sample_to_prompt(example)
        inputs = tokenizer(example['prompt'], truncation=False, padding=False)
        example['input_ids'] = inputs['input_ids']
        example['attention_mask'] = inputs['attention_mask']
        return example


    dataset = dataset.map(encode_race, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def _generate_config(tokenizer):

#     if tokenizer.__class__.__name__ == 'LlamaTokenizer':
#         eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
#         #eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
#     elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
#         eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
#     elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
#         eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
#     else:
#         raise NotImplementedError

    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
#     eos_token_id = [tokenizer.encode(_)[-1] for _ in ['\n']]
    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    # Follows Kuhn et al 2023 as Llama does not have CoQA
    # question_framing_ids = [[tokenizer(eos_token)['input_ids'][1]] for eos_token in question_framing_ids]
    question_framing_ids = [tokenizer(eos_token, add_special_tokens=False)['input_ids'] for eos_token in question_framing_ids]
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)

if __name__ == '__main__':
    import models
    dataset = get_dataset(models.load_tokenizer())