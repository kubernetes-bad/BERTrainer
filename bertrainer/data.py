from datasets import load_dataset, concatenate_datasets, IterableDataset, Dataset
from transformers import PreTrainedTokenizer


def load_and_tokenize_data(tokenizer: PreTrainedTokenizer, config) -> Dataset | IterableDataset:
    datasets = []
    for path in config['datasets']:
        if path.endswith('.jsonl') or path.endswith('.json'):
            dataset = load_dataset('json', data_files=path)
        else:
            dataset = load_dataset(path)

        if isinstance(dataset, dict):
            split = config.get('split', 'train')
            dataset = dataset[split]

        datasets.append(dataset)

    combined_dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    combined_dataset = combined_dataset.shuffle(seed=config['seed'])

    tokenized_datasets = combined_dataset.map(
        lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=config['max_seq_len']),
        batched=True
    )

    if 'train' not in tokenized_datasets.column_names and 'validation' not in tokenized_datasets.column_names:
        tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2, seed=config['seed'])
        tokenized_datasets['validation'] = tokenized_datasets.pop('test')

    return tokenized_datasets
