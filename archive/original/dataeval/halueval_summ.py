import os
import datasets
import _settings


def _save_dataset():
    """
    Downloads and saves the HaluEval summarization dataset locally.
    """
    save_path = f'{_settings.DATA_FOLDER}/HaluEval_summarization'
    
    if not os.path.exists(save_path):
        print(f"Local dataset not found. Downloading HaluEval (summarization) to {save_path}...")
        # Load the "summarization" subset from HaluEval
        data = datasets.load_dataset("pminervini/HaluEval", "summarization", split='data')
        
        data.save_to_disk(save_path)
    
    return save_path


def sample_to_prompt(sample, **kwargs):
    """
    Formats a sample from the dataset into a summarization prompt.
    """
    # A standard instruction-following prompt for summarization
    prompt = f"Summarize the following text: {sample['document']}Summary:"
    return prompt

def get_dataset(tokenizer):
    """
    Loads the processed HaluEval dataset, tokenizes it, and formats it for use.
    """
    dataset = datasets.load_from_disk(_save_dataset())
    id_map = {_['document']:str(i) for i, _ in enumerate(dataset)}

    def encode_halueval_summ(example):
        """Encodes a single example."""
        example['prompt'] = sample_to_prompt(example)
        #check for id , answer, question
        example['id'] = id_map[example['document']]
        example['question'] = example['document']
        example['answer'] = example['right_summary']
        inputs = tokenizer(example['prompt'], truncation=False, padding=False)
        
        example['input_ids'] = inputs['input_ids']
        example['attention_mask'] = inputs['attention_mask']
        return example

    dataset = dataset.map(encode_halueval_summ, batched=False, load_from_cache_file=False)
    
    # Set format for PyTorch, keeping all columns (like 'right_summary' for evaluation)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def _generate_config(tokenizer):
    """
    Generates stop tokens and bad words IDs for generation.
    """
    # Standard stop tokens: newline, period, and the tokenizer's official EOS token
    eos_token_id = [tokenizer.encode(_)[-1] for _ in ['\n']]
    eos_token_id += [tokenizer.eos_token_id]

    # Words to avoid repeating in the generated summary
    bad_words_list = ['Document:', '\nSummary:', 'Summary:', '\n']
    bad_words_ids = [tokenizer(word, add_special_tokens=False)['input_ids'] for word in bad_words_list]
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)

if __name__ == '__main__':
    import models
    dataset = get_dataset(models.load_tokenizer())