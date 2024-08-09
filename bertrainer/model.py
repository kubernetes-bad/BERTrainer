from transformers import AutoModelForSequenceClassification


def get_model(config):
    model_name = config['model_name']
    num_labels = config['num_labels']
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
