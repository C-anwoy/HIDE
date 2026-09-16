import os
import datasets
import _settings


def _save_dataset():
    """
    Downloads and saves the HaluEval summarization dataset locally.
    """
    save_path = f'{_settings.DATA_FOLDER}/HaluEval_dialogue'
    
    if not os.path.exists(save_path):
        print(f"Local dataset not found. Downloading HaluEval (dialogue) to {save_path}...")
        data = datasets.load_dataset("pminervini/HaluEval", "dialogue", split='data')
        
        data.save_to_disk(save_path)
    
    return save_path


def sample_to_prompt(sample, **kwargs):
    """
    Formats a sample from the dataset into a summarization prompt.
    """
    # A standard instruction-following prompt for summarization
    prompt = f"Answer from the following text: {sample['knowledge']}.\n{sample['dialogue_history']}[Assistant]:"
    return prompt

def get_dataset(tokenizer):
    """
    Loads the processed HaluEval dataset, tokenizes it, and formats it for use.
    """
    dataset = datasets.load_from_disk(_save_dataset())
    id_map = {_['dialogue_history']:str(i) for i, _ in enumerate(dataset)}

    def encode_halueval_dial(example):
        """Encodes a single example."""
        example['prompt'] = sample_to_prompt(example)
        #check for id , answer, question
        example['id'] = id_map[example['dialogue_history']]
        example['question'] = example['dialogue_history']
        example['answer'] = example['right_response']
        inputs = tokenizer(example['prompt'], truncation=False, padding=False)
        
        example['input_ids'] = inputs['input_ids']
        example['attention_mask'] = inputs['attention_mask']
        return example

    dataset = dataset.map(encode_halueval_dial, batched=False, load_from_cache_file=False)
    
    # Set format for PyTorch, keeping all columns (like 'right_summary' for evaluation)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def _generate_config(tokenizer):
    """
    Generates stop tokens and bad words IDs for generation.
    """
    # Standard stop tokens: newline, period, and the tokenizer's official EOS token
    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['\n', '.']]
    eos_token_id += [tokenizer.eos_token_id]

    # Words to avoid repeating in the generated summary
    bad_words_list = ['[Human]:', ' [Human]:', ' [Assistant]:', '[Assistant]:', '\n']
    bad_words_ids = [tokenizer(word, add_special_tokens=False)['input_ids'] for word in bad_words_list]
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)

if __name__ == '__main__':
    import models
    dataset = get_dataset(models.load_tokenizer())