import os
import json
import datasets
import pandas as pd
from datasets import Dataset
from hide import settings

def _save_dataset():
    save_path = f'{settings.data_folder()}/SQuAD'
    if not os.path.exists(save_path):
        with open('{}/dev-v2.0.json'.format(settings.data_folder()), 'r') as infile:
            data = json.load(infile)['data']
        dataset = {}
        dataset['story'] = []
        dataset['question'] = []
        dataset['answer'] = []
        dataset['additional_answers'] = []
        dataset['id'] = []
        for _data in data:
            paragraphs = _data['paragraphs']
            for (sample_id, sample) in enumerate(paragraphs):
                story = sample['context']
                questions = sample['qas']
                for (question_index, question) in enumerate(questions):
                    if question['is_impossible']:
                        continue
                    dataset['story'].append(story)
                    dataset['question'].append(question['question'])
                    dataset['answer'].append({'text': question['answers'][0]['text'], 'answer_start': question['answers'][0]['answer_start']})
                    dataset['id'].append(question['id'])
                    additional_answers_list = []
                    for i in range(len(question['answers'])):
                        additional_answers_list.append(question['answers'][i]['text'])
                    dataset['additional_answers'].append(additional_answers_list)
        dataset_df = pd.DataFrame.from_dict(dataset)
        dataset = Dataset.from_pandas(dataset_df)
        dataset.save_to_disk(save_path)
    return save_path

def get_dataset(tokenizer, split='validation'):
    dataset = datasets.load_from_disk(_save_dataset())

    def encode_coqa(example):
        example['answer'] = example['answer']['text']
        example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
        example.update(tokenizer(prompt, truncation=False, padding=False))
        return example
    dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset

def _generate_config(tokenizer):
    eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', '.']]
    eos_token_id += [tokenizer.eos_token_id]
    question_framing_ids = ['Question:', ' Question:', '\n', 'Answer:', ' Answer:', 'Q:']
    question_framing_ids = [tokenizer(eos_token, add_special_tokens=False)['input_ids'] for eos_token in question_framing_ids]
    return dict(eos_token_id=eos_token_id, bad_words_ids=question_framing_ids)
