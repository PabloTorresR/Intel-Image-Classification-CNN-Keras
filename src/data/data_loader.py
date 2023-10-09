from keras.utils import image_dataset_from_directory

from typing import Tuple, Optional
from tensorflow.keras import Sequential


def data_loader(
    data_dir: str,
    target_size: Tuple[int, int] = (150, 150),
    batch_size: int = 32,
    is_shuffle: bool = True,
    validation_split=0.2,
    data_augmentation_model: Optional[Sequential] = None,
):
    """
    Crea un cargador de datos para un modelo CNN.

    Args:
        data_dir (str): La ruta al directorio que contiene subdirectorios con imágenes.
        target_size (Tuple[int, int]): El tamaño al que se redimensionarán las imágenes (alto, ancho).
        batch_size (int): El tamaño del lote para entrenamiento.
        is_shuffle (bool): Si se debe mezclar el conjunto de datos.
        validation_split (float): Proporción de datos para validación.
        data_augmentation_model(Sequential, optional): Modelo de aumento de imágenes

    Returns:
        Any: Una instancia de ImageDataGenerator para cargar y aumentar imágenes.
    """

    data = image_dataset_from_directory(
        data_dir,
        image_size=target_size,
        batch_size=batch_size,
        shuffle=is_shuffle,
        seed=42,
        label_mode="categorical",
        validation_split=validation_split,
        subset="both",
    )

    if data_augmentation_model:
        print("augmenting images....")
        augmented_train = data_augmentation_model
        aug_train = data[0].map(lambda x, y: (augmented_train(x, training=True), y))
        return aug_train, data[1]

    return data


def data_loader_test(
    data_dir: str,
    target_size: Tuple[int, int] = (150, 150),
):
    """
    Crea un cargador de datos para test de un modelo CNN.

    Args:
        data_dir (str): La ruta al directorio que contiene subdirectorios con imágenes.
        target_size (Tuple[int, int]): El tamaño al que se redimensionarán las imágenes (alto, ancho).

    Returns:
        Any: Una instancia de ImageDataGenerator para cargar y aumentar imágenes.
    """

    data = image_dataset_from_directory(
        data_dir,
        image_size=target_size,
        label_mode="categorical",
    )

    return data
