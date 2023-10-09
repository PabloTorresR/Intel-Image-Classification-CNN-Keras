from src.config import CLASS_NAMES
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix


class ModelEvaluator:
    def __init__(self, saved_model: str, class_labels: list[str] = CLASS_NAMES):
        self.saved_model = saved_model
        self.class_labels = class_labels

    def __initialize_model__(self):
        return keras.models.load_model(self.saved_model)

    def evaluate(self, test_data):
        model = self.__initialize_model__()

        metrics = model.evaluate(test_data)  # type: ignore
        display_metrics = {"metrics": metrics}
        return metrics

    def evaluate_class_wise(self, test_data):
        test_labels = []
        test_images = []
        for batch in test_data:
            images, labels = batch
            for image, label in zip(images, labels):
                label = label.numpy()

                label_class_index = np.argmax(label)
                image_label = CLASS_NAMES[label_class_index]
                test_labels.append(image_label)
                test_images.append(image)

        test_preds = self.predict(np.array(test_images))
        cm = confusion_matrix(test_labels, test_preds)
        class_wise_accuracy = cm.diagonal() / cm.sum(axis=1)
        return class_wise_accuracy

    def predict(self, input_data):
        model = self.__initialize_model__()

        predictions = model.predict(input_data)  # type: ignore
        predicted_classes = []

        # Iterate through predictions and get the class with the highest probability
        for prediction in predictions:
            predicted_class_index = np.argmax(prediction)
            predicted_class = self.class_labels[predicted_class_index]
            predicted_classes.append(predicted_class)
        return predicted_classes

    def predict_one(self, input_data):
        model = self.__initialize_model__()

        return model.predict(input_data)  # type: ignore
