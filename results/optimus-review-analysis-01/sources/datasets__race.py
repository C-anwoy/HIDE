import os
import ast
import hashlib
import json
from pathlib import Path
import urllib.request
import datasets
import pandas as pd
from datasets import Dataset
from hide import settings

def parse_problems(value):
    problems = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(problems, list) or not all(isinstance(x, dict) for x in problems):
        raise ValueError('Expected a list of RACE question objects')
    return problems


def _save_dataset():
    save_path = f'{settings.data_folder()}/RACE'
    if not os.path.exists(save_path):
        raw_path = Path(settings.data_folder())/'race_high_test.jsonl'
        expected_sha = 'e0ff122cd99f1802cec824b824d576436b4c19c313744c7e1fe3f60c6b17f9d2'
        if not raw_path.exists():
            url = ('https://huggingface.co/datasets/EleutherAI/race/resolve/'
                   'e30efe648089df42c548e93faa9c5f1816e2c44f/race_high_test.jsonl')
            with urllib.request.urlopen(url, timeout=120) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != expected_sha:
                raise ValueError('RACE source checksum mismatch')
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
        raw = raw_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError('RACE source checksum mismatch; inspect the cache')
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
        dataset = {}
        dataset['story'] = []
        dataset['question'] = []
        dataset['answer'] = []
        dataset['id'] = []
        dataset['options'] = []
        for (sample_id, sample) in enumerate(data):
            story = sample['article']
            problems_str = sample['problems']
            questions = parse_problems(problems_str)
            for (question_index, question) in enumerate(questions):
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

def sample_to_prompt(sample, **kwargs):
    prompt = f"{sample['story']} Q: {sample['question']} \nOptions: {', '.join(sample['options'])} \nA:"
    return prompt

def get_dataset(tokenizer):
    dataset = datasets.load_from_disk(_save_dataset())

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
    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['.', '\n']]
    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [tokenizer(eos_token, add_special_tokens=False)['input_ids'] for eos_token in question_framing_ids]
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)
