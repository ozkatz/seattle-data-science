#!/usr/bin/env python
import sys
from pprint import pprint
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]

model_location = './models/is_alpaca.h5'

def predict(model, img_path):
    img = image.load_img(img_path, target_size=(244, 244))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    return prediction


if __name__ == '__main__':
    class_names = ['alpaca', 'not alpaca']
    model = tf.keras.models.load_model(model_location)
    result = dict(zip(class_names, predict(model, sys.argv[1])[0]))
    pprint(result)
