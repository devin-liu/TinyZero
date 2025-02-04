"""
Evaluation script for VizWiz VQA model.
"""

import logging
import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
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

def evaluate_vizwiz(
    checkpoint_dir: str = "outputs/vizwiz_vqa_test",
    data_dir: str = "data/vizwiz",
    batch_size: int = 8,
    max_test_samples: int = None,
):
    """
    Evaluate a trained VizWiz VQA model.
    """
    logger.info(f"Loading model from {checkpoint_dir}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    
    # Prepare test dataset
    logger.info("Preparing test dataset...")
    test_dataset = VizWizPPODataset(
        data_path=Path(data_dir) / "annotations/test.json",
        processor_name="microsoft/git-base-vqav2",  # Use the base model processor
        split="test",
    )
    
    # Training arguments for evaluation
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_eval_batch_size=batch_size,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
    )
    
    # Run evaluation
    logger.info("Running evaluation...")
    metrics = trainer.evaluate()
    
    # Print results
    logger.info("\nEvaluation Results:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

if __name__ == "__main__":
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "vizwiz"
    checkpoint_dir = base_dir / "outputs" / "vizwiz_vqa_test"
    
    evaluate_vizwiz(
        checkpoint_dir=str(checkpoint_dir),
        data_dir=str(data_dir),
    )
