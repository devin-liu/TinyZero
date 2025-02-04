"""
Simple training script for VizWiz VQA using HuggingFace Trainer.
"""

import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    TrainerCallback,
)
from examples.data_preprocess.vizwiz_ppo_dataset import VizWizPPODataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """Custom collate function to handle dynamic padding."""
    # Separate different features
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    
    # Get maximum sequence length in this batch
    max_length = max(ids.size(-1) for ids in input_ids)
    
    # Pad sequences to max length in batch
    input_ids_padded = []
    attention_mask_padded = []
    labels_padded = []
    
    for ids, mask, lab in zip(input_ids, attention_mask, labels):
        if ids.size(-1) < max_length:
            # Calculate padding
            pad_length = max_length - ids.size(-1)
            
            # Pad tensors
            ids_padded = torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)], dim=-1)
            mask_padded = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)], dim=-1)
            lab_padded = torch.cat([lab, torch.full((pad_length,), -100, dtype=lab.dtype)], dim=-1)
        else:
            ids_padded = ids
            mask_padded = mask
            lab_padded = lab
            
        input_ids_padded.append(ids_padded)
        attention_mask_padded.append(mask_padded)
        labels_padded.append(lab_padded)
    
    # Stack all tensors
    input_ids_padded = torch.stack(input_ids_padded)
    attention_mask_padded = torch.stack(attention_mask_padded)
    labels_padded = torch.stack(labels_padded)
    
    # Verify shapes match
    batch_size = input_ids_padded.size(0)
    assert input_ids_padded.size(0) == batch_size, f"input_ids batch size mismatch: {input_ids_padded.size(0)} vs {batch_size}"
    assert attention_mask_padded.size(0) == batch_size, f"attention_mask batch size mismatch: {attention_mask_padded.size(0)} vs {batch_size}"
    assert labels_padded.size(0) == batch_size, f"labels batch size mismatch: {labels_padded.size(0)} vs {batch_size}"
    assert pixel_values.size(0) == batch_size, f"pixel_values batch size mismatch: {pixel_values.size(0)} vs {batch_size}"
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "pixel_values": pixel_values,
        "labels": labels_padded,
    }

class ProgressCallback(TrainerCallback):
    """Callback to show training progress."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.logger.info(f"\n=== Starting epoch {state.epoch + 1}/{args.num_train_epochs} ===")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0 and len(state.log_history) > 0:  # Log every 10 steps
            self.logger.info(f"Step {state.global_step}/{state.max_steps} - Loss: {state.log_history[-1].get('loss', 'N/A')}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.info(f"Evaluation results at step {state.global_step}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"{key}: {value:.4f}")
                else:
                    self.logger.info(f"{key}: {value}")

def train_vizwiz(
    data_dir: str = "data/vizwiz",
    output_dir: str = "outputs/vizwiz_vqa_test",
    model_name: str = "microsoft/git-base-vqav2",
    batch_size: int = 4,
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    max_val_samples: int = 100,  # Limit validation samples for faster evaluation
    max_grad_norm: float = 1.0,  # Maximum gradient norm for clipping
    gradient_accumulation_steps: int = 4,  # Accumulate gradients over multiple steps
):
    """
    Train a vision-language model on the VizWiz dataset.
    """
    # Enable debug mode for transformers
    os.environ["TRANSFORMERS_VERBOSITY"] = "debug"
    
    # Set up logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{output_dir}/training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Prepare datasets
    logger.info("Loading datasets...")
    train_dataset = VizWizPPODataset(
        data_path=Path(data_dir) / "annotations/train.json",
        processor_name=model_name,
        split="train",
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    val_dataset = VizWizPPODataset(
        data_path=Path(data_dir) / "annotations/val.json",
        processor_name=model_name,
        split="val",
    )
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Limit validation samples
    if max_val_samples and max_val_samples < len(val_dataset):
        val_dataset = Subset(val_dataset, range(max_val_samples))
        logger.info(f"Limited val dataset size: {len(val_dataset)}")
    
    # Load model and processor
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Calculate total training steps for scheduler
    num_update_steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_ratio * max_train_steps)
    logger.info(f"Training steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"Total training steps: {max_train_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",  # Evaluate periodically
        eval_steps=100,  # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,  # Log every 10 steps
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/tensorboard reporting
        remove_unused_columns=False,  # Important: keep all columns
        label_names=["labels"],  # Specify label column
        fp16=False,  # Disable mixed precision for debugging
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,  # Use custom collate function
        callbacks=[ProgressCallback()],
        compute_metrics=None,  # Disable metrics computation for now
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "vizwiz"
    output_dir = base_dir / "outputs" / "vizwiz_vqa_test"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Start training
    train_vizwiz(
        data_dir="data/vizwiz",
        output_dir="outputs/vizwiz_vqa_test",
        model_name="microsoft/git-base-vqav2",
        batch_size=2,  # Reduced batch size
        num_epochs=1,  # Single epoch for testing
        learning_rate=5e-5,
        warmup_ratio=0.1,
        max_val_samples=10,  # Very few validation samples for quick testing
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,  # Reduced gradient accumulation
    )
