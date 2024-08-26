import numpy as np
import onnxruntime as ort
from image_process import generate_input
import random
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image as pil_image
import io, pathlib
from load_crop import *
from config import *

class ONNXModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        return self.session.run([self.output_name], {self.input_name: input_data})[0]


# Load the ONNX model
model = ONNXModel('model.onnx')


def infer_single(img_path):
    """Run inference on a single image with the ONNX model and return the predicted label."""
    model_input = generate_input(img_path)

    # ONNX inference
    raw_predictions = model.predict(model_input)

    # Get the index of the highest probability
    predicted_class_index = np.argmax(raw_predictions)

    # Map the index to the label
    predicted_label = LABEL_MAP[predicted_class_index]

    return predicted_label


def test_all_images(base_path):
    true_labels = []
    predicted_labels = []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                predicted_label = infer_single(img_path)
                true_labels.append(REVERSE_LABEL_MAP[folder])
                predicted_labels.append(REVERSE_LABEL_MAP[predicted_label])
                print(f"Image: {img_path}, True: {folder}, Predicted: {predicted_label}")

    return true_labels, predicted_labels


def plot_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_MAP.values(),
                yticklabels=LABEL_MAP.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()




if __name__ == "__main__":
    base_path = './simatic_photos'
    true_labels, predicted_labels = test_all_images(base_path)
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall Accuracy: {accuracy:.4f}")
    plot_confusion_matrix(true_labels, predicted_labels)
    print("Confusion matrix has been saved as 'confusion_matrix.png'")


