import os 
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path
import tensorflow as tf 
from tensorflow import keras
from cnnClassifier.entity.config_entity import TrainingModelConfig
from cnnClassifier import logger


class Training:
    def __init__(self,config: TrainingModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = keras.models.load_model(
            self.config.updated_model_path
        )
        

    def image_generater(self):
        self.train_ds = keras.utils.image_dataset_from_directory(
            directory = self.config.training_data,
            validation_split=0.2,
            subset="training",
            seed=123,
            batch_size = self.config.params_batch_size,
            image_size = self.config.params_image_size[:-1]
        )

        self.val_ds = keras.utils.image_dataset_from_directory(
            directory = self.config.training_data,
            validation_split=0.2,
            subset="validation",
            seed=123,
            batch_size = self.config.params_batch_size,
            image_size = self.config.params_image_size[:-1]
            )
        
        data_augmentation = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomContrast(0.2,input_shape = self.config.params_image_size),
            keras.layers.experimental.preprocessing.RandomRotation(0.2),
            keras.layers.experimental.preprocessing.RandomFlip(),
            keras.layers.experimental.preprocessing.RandomZoom(0.2),
        ])
        self.train_ds = self.train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

        
        self.train_ds = self.train_ds.map(self.process)
        self.val_ds = self.val_ds.map(self.process)

    def process(self,image,label):
        image = image/255
        label = tf.one_hot(label,self.config.params_classes)
        return image,label
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.model.fit(
            self.train_ds,
            validation_data = self.val_ds,
            epochs = self.config.params_epochs)
        
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )