from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml,create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig,PrepareBaseModelConfig,TrainingModelConfig,EvaluationConfig
import os

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = config.root_dir,
            base_model_path =  config.base_model_path,
            updated_model_path =  config.updated_model_path,
            params_image_size = self.params.IMAGE_SIZE,
            params_weights = self.params.WEIGHTS,
            params_include_top = self.params.INCLUDE_TOP,
            params_classes = self.params.CLASSES,
            params_learning_rate = self.params.LEARNING_RATE
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingModelConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model

        create_directories([training.root_dir])

        training_config = TrainingModelConfig(
            root_dir = Path(training.root_dir),
            updated_model_path = Path(prepare_base_model.updated_model_path),
            trained_model_path = training.trained_model_path,
            params_epochs = self.params.EPOCHS,
            params_augmentation = self.params.AUGMENTATION,
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE,
            params_classes = self.params.CLASSES,
            training_data = os.path.join(self.config.data_ingestion.unzip_dir,"kidney-ct-scan-image")
        )

        return training_config

    def get_model_evaluation_config(self) -> EvaluationConfig:
        training = self.config.training
        evaluation = self.config.evaluation

        evaluation_config = EvaluationConfig(
            path_of_model = training.trained_model_path,
            training_data = os.path.join(self.config.data_ingestion.unzip_dir,"kidney-ct-scan-image"),
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE,
            params_classes = self.params.CLASSES,
            all_params = self.params,
            mlflow_uri = evaluation.mlflow_uri
        )

        return evaluation_config

