from src.config import INPUT_DIM

from tensorflow import keras

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Rescaling,
    Dropout,
)
from tensorflow.keras.models import Model


def build_tunable_aumented_model(hp):
    # Descomentar si al final se optimiza esto
    # kernel_choices = hp.Int("filters", min_value=16, max_value=256, step=12)

    # Generamos un grid para funciones de activación
    activations = hp.Choice(
        "activation",
        values=[
            "relu",
            "sigmoid",
            "tanh",
            "selu",
            "elu",
        ],
    )

    # Generamos un grid para diferentes inicializadores de pesos
    kernel_init_choices = hp.Choice(
        "kernel_initializer",
        values=["glorot_uniform", "truncated_normal", "random_uniform", "ones"],
    )

    # Generamos un grid para diferentes regularizadores
    kernel_reg_choices = hp.Choice("kernel_regularizer", values=["l1_l2", "l1", "l2"])
    # Generamos un grid para valores de Dropout
    dropout_values = hp.Float("rate", min_value=0.1, max_value=0.6, step=0.1)

    inputs = Input(sahpe=INPUT_DIM)
    x = Rescaling(1.0 / 255)(inputs)

    x = Conv2D(
        32,
        kernel_size=(3, 3),
        activation=activations,
        kernel_initializer=kernel_init_choices,
        kernel_regularizer=kernel_reg_choices,
    )(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(rate=dropout_values)(x)

    x = Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        activation=activations,
        kernel_initializer=kernel_init_choices,
        kernel_regularizer=kernel_reg_choices,
    )(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(rate=dropout_values)(x)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        activation=activations,
        kernel_initializer=kernel_init_choices,
        kernel_regularizer=kernel_reg_choices,
    )(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(
        64,
        kernel_size=(3, 3),
        activation=activations,
        kernel_initializer=kernel_init_choices,
        kernel_regularizer=kernel_reg_choices,
    )(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(rate=dropout_values)(x)

    x = Flatten()(x)

    x = Dense(64, activation="relu")(x)

    outputs = Dense(6, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model
