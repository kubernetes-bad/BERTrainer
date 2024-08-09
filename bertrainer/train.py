import torch
from transformers import (
    Trainer, TrainingArguments, EarlyStoppingCallback,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, Adafactor
)
from torch.optim import Adam, AdamW
import wandb
from .data import load_and_tokenize_data
from .model import get_model
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, config):
    optimizer_name = config['optimizer']['name'].lower()
    if optimizer_name == 'adam':
        return Adam(model.parameters(),
                    lr=float(config['learning_rate']),
                    betas=tuple(float(beta) for beta in config['optimizer']['betas']),
                    eps=float(config['optimizer']['epsilon']))
    elif optimizer_name == 'adamw':
        return AdamW(model.parameters(),
                     lr=float(config['learning_rate']),
                     betas=tuple(float(beta) for beta in config['optimizer']['betas']),
                     eps=float(config['optimizer']['epsilon']))
    elif optimizer_name == 'adafactor':
        return Adafactor(model.parameters(),
                         lr=float(config['learning_rate']),
                         eps=(float(config['optimizer'].get('eps1', 1e-30)),
                              float(config['optimizer'].get('eps2', 1e-3))),
                         clip_threshold=float(config['optimizer'].get('clip_threshold', 1.0)),
                         decay_rate=float(config['optimizer'].get('decay_rate', -0.8)),
                         beta1=float(config['optimizer'].get('beta1')) if config['optimizer'].get('beta1') is not None else None,
                         weight_decay=float(config['optimizer'].get('weight_decay', 0.0)),
                         scale_parameter=config['optimizer'].get('scale_parameter', True),
                         relative_step=config['optimizer'].get('relative_step', True),
                         warmup_init=config['optimizer'].get('warmup_init', False))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer, num_training_steps, config):
    num_warmup_steps = config['warmup_steps']
    if config['lr_scheduler'] == 'linear':
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    elif config['lr_scheduler'] == 'cosine':
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config['lr_scheduler']}")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_model(config):
    set_seed(config['seed'])
    device = get_device()
    print(f"Using device: {device}")

    model = get_model(config)
    model.to(device)

    tokenized_datasets = load_and_tokenize_data(config)

    optimizer = get_optimizer(model, config)

    num_update_steps_per_epoch = len(tokenized_datasets["train"]) // config['train_batch_size']
    num_training_steps = config['num_epochs'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(optimizer, num_training_steps, config)

    if device.type == 'cuda':
        fp16 = config.get('fp16', True)
        bf16 = config.get('bf16', False)
    else:
        # both MPS and CPU can't do those
        fp16 = False
        bf16 = False

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        warmup_steps=config['warmup_steps'],
        weight_decay=config.get('weight_decay', 0),
        logging_dir=config['logging_dir'],
        logging_steps=10,
        report_to="wandb" if config.get('use_wandb', False) else None,
        eval_strategy=config['eval_strategy'],
        eval_steps=config['eval_steps'] if config['eval_strategy'] == "steps" else None,
        save_steps=config['eval_steps'] if config['eval_strategy'] == "steps" else None,
        bf16=bf16,
        fp16=fp16,
        learning_rate=config['learning_rate'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        eval_accumulation_steps=config['eval_accumulation_steps'],
        seed=config['seed'],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])],
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()
    model.save_pretrained(f"{config['output_dir']}/final")


def train(config):
    if config.get('use_wandb', False):
        wandb.init(project=config.get('wandb_project', 'text-classification'))

    if config.get('use_wandb_sweep', False):
        sweep_config = config['wandb_sweep_config']
        sweep_id = wandb.sweep(sweep_config, project=config.get('wandb_project', 'text-classification'))
        wandb.agent(sweep_id, lambda: train_model(config), count=config.get('wandb_sweep_runs', 5))
    else:
        train_model(config)


if __name__ == "__main__":
    import sys
    from .config import load_config

    if len(sys.argv) != 2:
        print("Usage: python3 -m bertrainer.train <yo_config_file>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    train(cfg)
