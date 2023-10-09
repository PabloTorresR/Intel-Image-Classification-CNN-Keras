from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.evaluator import ModelEvaluator


def plot_model_metrics(test_data, class_names: list[str], models: Dict[str, str]):
    """
    Grafica métricas de precisión por clase para varios modelos.

    Parámetros:
        class_names (list): Una lista de nombres de clases para las cuales se calculan las métricas de precisión.
        models (dict): Un diccionario donde las claves son nombres de modelos y los valores son rutas o configuraciones de modelos.
        test_data: Los datos de prueba utilizados para la evaluación.

    Esta función toma una lista de nombres de clases, un diccionario de nombres de modelos y datos de prueba y genera una gráfica que muestra las métricas de precisión por clase para cada modelo.
    """

    x = np.arange(len(class_names))
    sns.set(style="whitegrid")
    sns.color_palette("rocket")

    plt.figure(figsize=(12, 6))  # Ajusta el tamaño de la figura

    for model_name, model_value in models.items():
        model_evaluator = ModelEvaluator(model_value)
        model_metrics = model_evaluator.evaluate_class_wise(test_data)
        sns.lineplot(x=x, y=model_metrics, label=model_name)

    plt.xticks(x, class_names, rotation=45)
    plt.xlabel("Clases")
    plt.ylabel("Precisión")
    plt.legend()
    plt.grid(True)

    plt.show()
