"""Configuration for VizWiz VQA training using PPO."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from verl.core.config import Config
from verl.trainer.config import TrainerConfig
from verl.data.config import DataConfig
from verl.models.config import ModelConfig
from verl.algorithm.config import AlgorithmConfig

@dataclass
class VizWizDataConfig(DataConfig):
    """Configuration for VizWiz dataset."""
    train_files: List[str]
    val_files: List[str]
    train_batch_size: int = 32
    val_batch_size: int = 64
    max_prompt_length: int = 256
    max_response_length: int = 128
    image_size: int = 480  # GIT model's default image size
    num_workers: int = 4
    dataset_type: str = "vizwiz"  # Specify that we're using VizWiz dataset
    dataset_config: Dict[str, Any] = field(default_factory=lambda: {
        "processor_name": "microsoft/git-base-vqav2",
        "max_prompt_length": 256,
        "max_response_length": 128,
        "image_size": 480,
    })

@dataclass
class VizWizModelConfig(ModelConfig):
    """Configuration for VizWiz model."""
    path: str
    vision_tower: Optional[str] = None
    use_flash_attention: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
@dataclass
class VizWizConfig(Config):
    """Configuration for VizWiz VQA training."""
    data: VizWizDataConfig
    model: VizWizModelConfig
    trainer: TrainerConfig
    algorithm: AlgorithmConfig

    def __post_init__(self):
        super().__post_init__()
        
        # Set default values for VizWiz specific configuration
        if not hasattr(self.data, 'image_size'):
            self.data.image_size = 480
            
        # Update model configuration for vision-language tasks
        if not hasattr(self.model, 'vision_tower'):
            self.model.vision_tower = None
