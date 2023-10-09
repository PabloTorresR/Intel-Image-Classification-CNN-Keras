from src.config import INPUT_DIM

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Rescaling,
)
from tensorflow.keras.models import Model


def create_simple_cnn(num_classes: int):
    # Input layer
    inputs = Input(shape=INPUT_DIM)
    x = Rescaling(1.0 / 255)(inputs)

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten layer
    x = Flatten()(x)

    # Dense layers
    x = Dense(256, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model
