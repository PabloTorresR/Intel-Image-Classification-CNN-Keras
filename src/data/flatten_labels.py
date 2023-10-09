from src.config import CLASS_NAMES
import numpy as np


def flatten_ds_labels(data):
    """Extrae las etiquetas de un ds en batch"""
    test_labels = []
    for batch in data:
        labels = batch[1]
        labels = labels.numpy()
        for label in labels:
            label_class_index = np.argmax(label)
            image_label = CLASS_NAMES[label_class_index]
            test_labels.append(image_label)
    return test_labels
