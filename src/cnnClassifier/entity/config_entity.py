from dataclasses import dataclass 
from pathlib import Path 

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_model_path: Path
    params_image_size: list
    params_weights: str
    params_include_top: bool
    params_classes: int
    params_learning_rate: float

@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    updated_model_path: Path
    trained_model_path: Path
    params_epochs: int
    params_augmentation: bool
    training_data: Path
    params_image_size: list
    params_batch_size: list
    params_classes: int