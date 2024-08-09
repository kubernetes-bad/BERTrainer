from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def load_and_tokenize_data(config):
    datasets = []
    for path in config['datasets']:
        dataset = load_dataset('json', data_files=path)

        if isinstance(dataset, dict):
            split = config.get('split', 'train')
            dataset = dataset[split]

        datasets.append(dataset)

    combined_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    combined_dataset = combined_dataset.shuffle(seed=config['seed'])

    # TODO: fast tokenizer for roberta is unpredictable with spaces
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=False, clean_up_tokenization_spaces=True)
    tokenized_datasets = combined_dataset.map(
        lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=config['max_seq_len']),
        batched=True
    )

    if 'train' not in tokenized_datasets.column_names and 'validation' not in tokenized_datasets.column_names:
        tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2, seed=config['seed'])
        tokenized_datasets['validation'] = tokenized_datasets.pop('test')

    return tokenized_datasets
