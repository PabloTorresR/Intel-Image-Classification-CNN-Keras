from typing import Dict
import pandas as pd
from tabulate import tabulate
from src.models.evaluator import ModelEvaluator


def get_results_table(test_data, models: Dict[str, str]):
    results_table = []

    for model_name, model_value in models.items():
        model_evaluator = ModelEvaluator(model_value)
        results = model_evaluator.evaluate(test_data) or None  # type: ignore

        if results and len(results) > 1:
            model_metrics = {
                "Model Name": model_name,
                "Metric Value": results[1],
            }
        else:
            model_metrics = {"Model Name": model_name, "Accuracy Value": None}

        results_table.append(model_metrics)

    df = pd.DataFrame(results_table)

    table = tabulate(df, headers="keys", tablefmt="pretty", showindex=False)

    return table
