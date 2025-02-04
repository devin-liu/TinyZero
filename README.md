# TinyZero
![image](cover.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own 

You can experience the Ahah moment yourself for < $30 

Twitter thread: https://x.com/jiayi_pirate/status/1882839370505621655

Full experiment log: https://wandb.ai/jiayipan/TinyZero

Paper's on it's way!

## Installation

We use Poetry for dependency management. First, install Poetry if you haven't already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then install the project dependencies:

```bash
# Clone the repository
git clone https://github.com/Jiayi-Pan/TinyZero.git
cd TinyZero

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

Note: For Apple Silicon (M1/M2) users, if you encounter issues with the `_lzma` module, you may need to reinstall Python with proper build flags:

```bash
# Install required system libraries
brew install xz zlib

# Reinstall Python with proper build flags
LDFLAGS="-L/opt/homebrew/opt/zlib/lib -L/opt/homebrew/opt/xz/lib" \
CPPFLAGS="-I/opt/homebrew/opt/zlib/include -I/opt/homebrew/opt/xz/include" \
pyenv install --force 3.9.16
```

## Countdown task

**Data Preparation**
```bash
# Make sure you're in the Poetry environment
poetry run python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

### Run Training

For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script, and checkout the discussion [here](https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643)

**Single GPU**

Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```bash
export N_GPUS=1
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

poetry run bash ./scripts/train_tiny_zero.sh
```

**3B+ model**
In this case, the base model is able to develop sophisticated reasoning skills.
```bash
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

poetry run bash ./scripts/train_tiny_zero.sh
```

### Instruct Ablation
We experiment with QWen-2.5-3B Instruct too.
**Data Preparation**
To follow chat template, we need to reprocess the data:
```bash
poetry run python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**
```bash
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

poetry run bash ./scripts/train_tiny_zero.sh
```

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
