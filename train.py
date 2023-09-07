#!/usr/bin/env python
import tensorflow as tf
from keras import Model, Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras import layers

input_location = './input'
model_location = './models/is_alpaca.h5'

selection_params = dict(
    seed = 123,
    validation_split = 0.2,
    batch_size = 32,
    image_size = (244 ,244),
)

def get_ds(subset: str) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        input_location, 
        subset=subset, **selection_params)


def build_model() -> Model:
    model = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2)
    ])
    model.compile(optimizer='adam', 
                loss=SparseCategoricalCrossentropy(from_logits=True), 
                metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Fit and save
    model = build_model()
    model.fit(
        get_ds('training'), 
        validation_data=get_ds('validation'), 
        epochs=3)
    model.save(model_location)

