from tensorflow.keras import Sequential, layers
from src.config import INPUT_DIM


def get_sequential_augmentation():
    model = Sequential(
        [
            layers.experimental.preprocessing.RandomFlip(
                "horizontal", input_shape=(INPUT_DIM)
            ),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
            layers.experimental.preprocessing.RandomContrast(factor=(0.8, 1.4)),
            layers.experimental.preprocessing.RandomTranslation(
                height_factor=0.1, width_factor=0.1
            ),
        ]
    )
    return model
