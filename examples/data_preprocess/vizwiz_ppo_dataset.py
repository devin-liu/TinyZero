"""
VizWiz dataset for PPO training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

class VizWizPPODataset(torch.utils.data.Dataset):
    """
    Dataset class for VizWiz VQA using PPO.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        processor_name: str = "microsoft/git-base-vqav2",
        split: str = "train",
        max_length: int = 512,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the annotations file
            processor_name: Name of the processor to use
            split: Dataset split (train, val)
            max_length: Maximum sequence length for text inputs
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_length = max_length
        
        # Load processor
        logger.info(f"Loading processor: {processor_name}")
        self.processor = AutoProcessor.from_pretrained(processor_name)
        
        # Load annotations
        logger.info(f"Loading annotations from {self.data_path}")
        with open(self.data_path, "r") as f:
            self.annotations = json.load(f)
        logger.info(f"Loaded {len(self.annotations)} annotations")
        
        # Get base image directory
        self.image_dir = self.data_path.parent.parent / "images" / split / split
        logger.info(f"Image directory: {self.image_dir}")
        
        # Verify images exist
        self._verify_images()
    
    def _verify_images(self):
        """Verify that all images exist and are readable."""
        logger.info("Verifying images...")
        valid_annotations = []
        for ann in self.annotations:
            image_path = self.image_dir / ann["image"]
            if image_path.exists():
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                    valid_annotations.append(ann)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")
            else:
                logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"Found {len(valid_annotations)} valid images out of {len(self.annotations)}")
        self.annotations = valid_annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def _create_default_item(self):
        """Create a default item with zero tensors."""
        return {
            "input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "pixel_values": torch.zeros(3, 224, 224, dtype=torch.float),
            "labels": torch.zeros(self.max_length, dtype=torch.long),
        }
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns:
            Dict containing input_ids, attention_mask, pixel_values, and labels
        """
        try:
            ann = self.annotations[idx]
            
            # Load and preprocess image
            image_path = self.image_dir / ann["image"]
            try:
                logger.debug(f"Loading image: {image_path}")
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                return self._create_default_item()
            
            # Prepare text inputs
            question = ann["question"]
            answers = ann.get("answers", [{"answer": "no answer available"}])
            answer = answers[0]["answer"]  # Take first answer
            
            logger.debug(f"Processing item {idx}")
            logger.debug(f"Question: {question}")
            logger.debug(f"Answer: {answer}")
            
            try:
                # Process question and image together
                question_inputs = self.processor(
                    images=image,
                    text=f"Question: {question}",
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                )
                
                # Process answer text
                answer_inputs = self.processor.tokenizer(
                    f"Answer: {answer}",
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                )
                
                # Remove batch dimension
                question_inputs = {k: v.squeeze(0) for k, v in question_inputs.items()}
                answer_inputs = {k: v.squeeze(0) for k, v in answer_inputs.items()}
                
                # Ensure labels have same length as input_ids by padding
                if answer_inputs["input_ids"].size(0) < question_inputs["input_ids"].size(0):
                    pad_length = question_inputs["input_ids"].size(0) - answer_inputs["input_ids"].size(0)
                    answer_inputs["input_ids"] = torch.cat([
                        answer_inputs["input_ids"],
                        torch.full((pad_length,), self.processor.tokenizer.pad_token_id, dtype=torch.long)
                    ])
                
                # Create final inputs with matched lengths
                inputs = {
                    "input_ids": question_inputs["input_ids"],
                    "attention_mask": question_inputs["attention_mask"],
                    "pixel_values": question_inputs["pixel_values"],
                    "labels": answer_inputs["input_ids"],
                }
                
                # Verify tensor shapes
                assert inputs["input_ids"].size(0) == inputs["labels"].size(0), \
                    f"Mismatched lengths: input_ids={inputs['input_ids'].size(0)}, labels={inputs['labels'].size(0)}"
                
                return inputs
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                return self._create_default_item()
        except Exception as e:
            logger.error(f"Error in __getitem__ for idx {idx}: {e}")
            return self._create_default_item()

def prepare_vizwiz_ppo_dataset(
    data_path: Union[str, Path],
    processor_name: str = "microsoft/git-base-vqav2",
    split: str = "train",
) -> VizWizPPODataset:
    """
    Prepare VizWiz dataset for PPO training.
    
    Args:
        data_path: Path to the annotations file
        processor_name: Name of the vision-language processor to use
        split: Dataset split to prepare
        
    Returns:
        Prepared VizWiz dataset
    """
    return VizWizPPODataset(
        data_path=data_path,
        processor_name=processor_name,
        split=split,
    )
