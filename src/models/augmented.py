from src.config import INPUT_DIM

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Rescaling,
)
from tensorflow.keras.models import Model


def create_augmented_model(num_classes: int):
    inputs = Input(shape=INPUT_DIM)
    x = Rescaling(1.0 / 255)(inputs)

    x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(64, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
