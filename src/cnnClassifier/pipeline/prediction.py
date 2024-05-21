import tensorflow as tf 
from tensorflow import keras 
import os
from keras.preprocessing import image
import numpy as np



class PredictionPipeline:
    def __init__(self):
        pass

    def prediction(self,image_path,image_size):
        model = keras.models.load_model(os.path.join('artifacts','training','model.h5'))
        image = keras.utils.load_img(image_path,target_size=image_size)
        input_arr = keras.utils.img_to_array(image)
        input_arr = input_arr / 255
        input_arr = np.expand_dims(input_arr, axis = 0)
        p = model.predict(input_arr)
        print(p)
        result = np.argmax(p,axis=1)
        
        

        return result
    
    