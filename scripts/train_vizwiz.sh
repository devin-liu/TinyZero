#!/bin/bash

# Configuration
export CUDA_VISIBLE_DEVICES=0  # Use single GPU for testing

# Create necessary directories
mkdir -p outputs/vizwiz_vqa_ppo_test

# Run training with small batch size using Poetry and test config
poetry run python3 -m verl.trainer.main_ppo \
data=vizwiz_test 2>&1 | tee outputs/vizwiz_vqa_ppo_test/training.log
