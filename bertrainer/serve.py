import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from litserve import LitAPI, LitServer

from .utils import flatten
from .config import load_config


class BERTLitAPI(LitAPI):
    def __init__(self, config_path: str) -> None:
        self.tokenizer = None
        self.model = None
        self.config = load_config(config_path)

    def setup(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            use_fast=False,
            clean_up_tokenization_spaces=True,
        )

        model_path = f"{self.config['output_dir']}/final"
        if not os.path.exists(model_path) or not os.path.exists(f"{model_path}/config.json"):
            raise FileNotFoundError(f"Model config.json file not found at {model_path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=self.config['num_labels']
        )

        self.model.to(device)
        self.model.eval()

    def decode_request(self, request, **kwargs):
        inputs = self.tokenizer(
            request["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_seq_len'])
        return inputs

    def predict(self, inputs, **kwargs):
        with torch.no_grad():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        return outputs.logits

    def encode_response(self, logits, **kwargs):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        response = {f"class_{i}": prob.item() for i, prob in enumerate(probabilities[0])}
        return response


def serve(config_path):
    api = BERTLitAPI(config_path)
    api.config_path = config_path
    config = load_config(config_path)
    server = LitServer(api, accelerator='auto', devices=1)
    print("Running on " + ", ".join([worker for worker in flatten(server.workers)]))
    server.run(
        port=config.get('serve_port', 8000),
        generate_client_file=False,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 -m trainer.serve <config_file>")
        sys.exit(1)

    serve(sys.argv[1])
