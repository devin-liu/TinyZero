"""
Data preprocessing for the VizWiz Visual Question Answering dataset.
VizWiz is a dataset containing real-world images taken by blind people along with questions about these images
and their corresponding answers.

Dataset reference: https://vizwiz.org/tasks-and-datasets/vqa/
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

class VizWizDataset(Dataset):
    """Dataset class for VizWiz Visual Question Answering."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        processor_name: str = "microsoft/git-base-vqav2",
        max_length: int = 512,
    ):
        """
        Initialize VizWiz dataset.
        
        Args:
            data_dir: Directory containing the VizWiz dataset
            split: Dataset split ('train', 'val', or 'test')
            processor_name: Name of the vision-language processor to use
            max_length: Maximum length for text tokenization
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        
        # Load annotations
        ann_file = self.data_dir / "annotations" / f"{split}.json"
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(processor_name)
        
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example to fetch
            
        Returns:
            Dictionary containing processed inputs for the model
        """
        ann = self.annotations[idx]
        image_id = ann["image"]
        question = ann["question"]
        
        # Load and preprocess image
        image_path = self.data_dir / "images" / self.split / self.split / image_id
        image = Image.open(image_path).convert("RGB")
        
        # Get answer if available (not available for test split)
        answer = None
        if self.split != "test" and "answers" in ann:
            answer = ann["answers"][0]["answer"]  # Using first answer
            
        # Process inputs
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        if answer:
            # Process answer if available
            answer_tokens = self.processor(
                text=answer,
                return_tensors="pt",
                padding="max_length",
                max_length=32,  # Shorter max_length for answers
                truncation=True,
            )
            inputs["labels"] = answer_tokens.input_ids.squeeze(0)
            
        return inputs

def prepare_vizwiz_dataset(
    data_dir: Union[str, Path],
    split: str = "train",
    processor_name: str = "microsoft/git-base-vqav2",
) -> VizWizDataset:
    """
    Prepare VizWiz dataset for training or evaluation.
    
    Args:
        data_dir: Directory containing the VizWiz dataset
        split: Dataset split to prepare
        processor_name: Name of the vision-language processor to use
        
    Returns:
        Prepared VizWiz dataset
    """
    return VizWizDataset(
        data_dir=data_dir,
        split=split,
        processor_name=processor_name,
    )

if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent.parent.parent / "data" / "vizwiz"
    dataset = prepare_vizwiz_dataset(data_dir=data_dir, split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Example of accessing a single item
    sample = dataset[0]
    print("Keys in processed sample:", sample.keys())
