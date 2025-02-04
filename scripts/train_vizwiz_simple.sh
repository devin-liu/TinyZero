#!/bin/bash

# Configuration
export CUDA_VISIBLE_DEVICES=0  # Use single GPU for testing

# Create necessary directories
mkdir -p outputs/vizwiz_vqa_test

# Get the poetry executable path
POETRY_PATH=$(which poetry || echo "$HOME/.local/bin/poetry")

# Check if poetry exists
if [ ! -f "$POETRY_PATH" ]; then
    echo "Poetry not found. Installing poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    POETRY_PATH="$HOME/.local/bin/poetry"
fi

# Install dependencies if needed
"$POETRY_PATH" install

# Run training with small batch size using Poetry
"$POETRY_PATH" run python3 examples/train_vizwiz_simple.py 2>&1 | tee outputs/vizwiz_vqa_test/training.log
