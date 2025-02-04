"""
Fine-tune a vision-language model on the VizWiz Visual Question Answering dataset.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from examples.data_preprocess.vizwiz import prepare_vizwiz_dataset

def train_vizwiz(
    data_dir: str = "data/vizwiz",
    output_dir: str = "outputs/vizwiz_vqa",
    model_name: str = "microsoft/git-base-vqav2",
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
):
    """
    Fine-tune a vision-language model on the VizWiz dataset.
    
    Args:
        data_dir: Directory containing the VizWiz dataset
        output_dir: Directory to save model checkpoints and logs
        model_name: Name of the pre-trained model to use
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        warmup_ratio: Ratio of steps for learning rate warmup
    """
    # Prepare datasets
    print("Loading datasets...")
    train_dataset = prepare_vizwiz_dataset(data_dir, split="train")
    val_dataset = prepare_vizwiz_dataset(data_dir, split="val")
    
    # Load model and processor
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # Keep only the last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        remove_unused_columns=False,  # Important for vision-language models
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "vizwiz"
    output_dir = base_dir / "outputs" / "vizwiz_vqa"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    train_vizwiz(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        batch_size=4,  # Small batch size for testing
        num_epochs=3,  # Just a few epochs for testing
    )
