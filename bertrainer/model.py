from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer


def get_model(config):
    model_name = config['model_name']
    num_labels = config['num_labels']

    extra_args = {}

    if 'labels' in config:
        id2label = {int(k): v for k, v in config['labels'].items()}
        extra_args['id2label'] = id2label
        extra_args['label2id'] = {v: k for k, v in id2label.items()}

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        **extra_args
    )


def get_tokenizer(config) -> PreTrainedTokenizer:
    # TODO: fast tokenizer for roberta is unpredictable with spaces
    return AutoTokenizer.from_pretrained(config['model_name'], use_fast=False, clean_up_tokenization_spaces=True)
