# BERTrainer

BERTrainer is designed to make your life easier when training text classification models.

If you could handle [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), you can handle BERTrainer too.

## Features

- Supports BERT, DeBERTa, RoBERTa (probably more)
- Yaml configs, yay!
- CUDA, MPS, and CPU are supported
- Weights & Biases Sweeps 🙌
- Multiple datasets in one training, shuffled

## Installation

To get started, install it using pip:


```bash
git clone https://github.com/kubernetes-bad/BERTrainer
cd BERTrainer
pip3 install -e .
```

Or use Docker:

```
docker run -it \
  -e WANDB_API_KEY=abcdef00008888 \
  -v /path/to/config.yaml:/config.yaml \
  -v /path/to/output/:/output \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  ghcr.io/kubernetes-bad/bertrainer /config.yaml
```

## Usage

Using BERTrainer is easy, the design is very human. Just follow these steps:

1. Create a configuration file (e.g., `config.yml`) specifying your model, dataset, and training settings. Check out the [example configurations](./examples) for inspiration.

2. Run the trainer with your configuration file:
    
    ```bash
    python3 -m bertrainer.train config.yml
    ```

3. Sit back, watch the graphs, and let the trainer do its magic! ✨

4. Once the training is complete, you'll find your trained model in the specified output directory.

Happy training! 🎓✨
