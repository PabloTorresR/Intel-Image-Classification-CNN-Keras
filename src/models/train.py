from src.models.simple import create_simple_cnn
from src.models.augmented import create_augmented_model
from src.models.custom_xception import create_custom_xception

import comet_ml
from enum import Enum
from tensorflow import keras
from typing import Optional


class ModelConfiguration(Enum):
    SIMPLE = "simple"
    AUGMENTED = "augmented"
    TUNABLE = "tunable"
    XCEPTION_CUSTOM = "xception_custom"


def get_model(
    model_config: ModelConfiguration,
    classes_predicted: int,
    saved_model_name: Optional[str],
    n_layers_to_train: Optional[int],
):
    """
    Obtiene un modelo de acuerdo a la configuración especificada.

    Args:
        model_config (ModelConfiguration): Enum que especifica la configuración del modelo.
        classes_predicted (int): Número de clases a predecir.
        saved_model_name (str, optional): Nombre del modelo a cargar si se utiliza la configuración 'TUNABLE'.
        n_layers_to_train (int, optional): Número de layers a entrenar para el caso de uso de que el modelo sea de fine-tunning.

    Returns:
        keras.Model: Modelo de TensorFlow.
    """

    if model_config == ModelConfiguration.SIMPLE:
        return create_simple_cnn(classes_predicted)
    elif model_config == ModelConfiguration.AUGMENTED:
        return create_augmented_model(classes_predicted)
    elif model_config == ModelConfiguration.TUNABLE:
        return keras.models.load_model(f"../models/{saved_model_name}")
    elif model_config == ModelConfiguration.XCEPTION_CUSTOM:
        return (
            create_custom_xception(
                num_classes=classes_predicted, num_last_frozen_layers=n_layers_to_train
            )
            if n_layers_to_train
            else create_custom_xception(classes_predicted)
        )
    else:
        return create_simple_cnn(classes_predicted)


class ModelTrainer:
    """
    Inicializa un objeto ModelTrainer.

    Args:
        model_name (str): Nombre del modelo.
        model_config (ModelConfiguration): Enum que especifica la configuración del modelo.
        classes_predicted (int): Número de clases a predecir.
        n_layers_to_train (int, optional): Número de layers a entrenar para el caso de uso de que el modelo sea de fine-tunning.

    """

    def __init__(
        self,
        model_name: str,
        model_config: ModelConfiguration,
        classes_predicted: int,
    ):
        self.model_name = model_name
        self.model_config = model_config
        self.classes_predicted = classes_predicted

    def __initialize_experiment__(self):
        """
        Inicializa un experimento de Comet.ml.
        """
        experiment = comet_ml.Experiment(
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
            api_key="*****",  # Se ha eliminado la key del alumno
            project_name="*****",
            workspace="*****",
        )
        return experiment

    def __create_callbacks__(self):
        earlystopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=0, mode="min"
        )
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath="bestvalue", verbose=0, save_best_only=True
        )
        return [checkpointer, earlystopping]

    def __get_save_folder_name__(self):
        return f"../models/{self.model_name}"

    def train(
        self,
        train_data,
        validation_data,
        n_layers_to_train: Optional[int] = None,
        epochs=25,
    ):
        """
        Entrena el modelo especificado.

        Args:
            train_data: Datos de entrenamiento.
            validation_data: Datos de validación.
            epochs (int, optional): Número de epochs de entrenamiento.

        Returns:
            None
        """
        model = get_model(
            self.model_config,
            self.classes_predicted,
            self.model_name,
            n_layers_to_train,
        )
        if not model:
            return

        experiment = self.__initialize_experiment__()
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        model.compile(
            optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        with experiment.train():
            model.fit(
                train_data,
                validation_data=validation_data,
                batch_size=128,
                epochs=epochs,
                callbacks=self.__create_callbacks__(),
            )
        experiment.end()
        model_save_path = self.__get_save_folder_name__()
        model.save(model_save_path)
