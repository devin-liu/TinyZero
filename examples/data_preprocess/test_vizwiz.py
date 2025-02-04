"""Test script for VizWiz dataset loader."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from vizwiz import prepare_vizwiz_dataset

def test_vizwiz_sample():
    # Initialize dataset
    data_dir = Path(__file__).parent.parent.parent / "data" / "vizwiz"
    dataset = prepare_vizwiz_dataset(data_dir=data_dir, split="train")
    
    # Get a sample
    idx = 0  # Let's look at the first item
    sample = dataset[idx]
    
    # Print sample information
    print("\nSample Information:")
    print("Input IDs shape:", sample['input_ids'].shape)
    print("Attention Mask shape:", sample['attention_mask'].shape)
    print("Pixel Values shape:", sample['pixel_values'].shape)
    if 'labels' in sample:
        print("Labels shape:", sample['labels'].shape)
    
    # Load annotations to get original data
    with open(data_dir / "annotations" / "train.json", 'r') as f:
        ann = json.load(f)[idx]
    
    # Get the original image, question and answers
    image_id = ann['image']
    question = ann['question']
    answers = [a['answer'] for a in ann['answers']]
    
    # Load and display the image
    image_path = data_dir / "images" / "train" / "train" / image_id
    image = Image.open(image_path)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Question: {question}\nAnswers: {', '.join(answers)}", wrap=True)
    
    # Save the plot
    plt.savefig(data_dir / "sample_visualization.png", bbox_inches='tight', dpi=300)
    print("\nVisualization saved as 'sample_visualization.png'")
    
    # Print additional information
    print("\nQuestion:", question)
    print("Answers:", answers)
    print("\nProcessed tensors are ready for model input")

if __name__ == "__main__":
    test_vizwiz_sample()
