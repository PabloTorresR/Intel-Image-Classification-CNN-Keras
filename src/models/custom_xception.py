from src.config import INPUT_DIM

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from typing import Tuple


def get_base_xception(input_shape: Tuple[int, int, int], num_last_frozen_layers: int):
    """
    Crea y configura el modelo base Xception para transferencia de aprendizaje con control de ajuste fino.

    Args:
        input_shape (tuple): La forma de las imágenes de entrada en formato (alto, ancho, canales).
        num_last_frozen_layers (int): El número de últimas capas a descongelar para ajuste fino.
    Returns:
        keras.Model: Modelo de TensorFlow.
    """
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    # Congelar todas las capas
    base_model.trainable = False

    # Descongelar ultimas N capas
    for layer in base_model.layers[-num_last_frozen_layers:]:
        layer.trainable = True
    return base_model


def create_custom_xception(num_classes: int, num_last_frozen_layers: int = 0):
    """
    Crea un modelo de clasificación personalizado basado en el modelo base Xception con control de ajuste fino.

    Args:
        num_classes (int): El número de clases de salida para la clasificación.
        num_last_frozen_layers (int): El número de últimas capas a descongelar para el ajuste fino.

    Returns:
        tensorflow.keras.Model: Modelo de TensorFlow.
    """
    base_xception_model = get_base_xception(INPUT_DIM, num_last_frozen_layers)
    x = Flatten()(base_xception_model.output)

    x = Dense(128, activation="relu")(x)

    x = Dense(64, activation="relu")(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    custom_model = Model(inputs=base_xception_model.input, outputs=outputs)

    return custom_model
