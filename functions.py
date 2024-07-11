import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def make_image_tensor(subdir):
    # Append resized cv2 images in the specified subdirectory to an array
    image_array = []
    for root, dirs, files in os.walk(os.path.abspath(subdir)):
        for file in files:
            image = cv2.imread(os.path.join(root, file))
            image = cv2.resize(image, (75, 75))
            image_array.append(image)

    return image_array


def show_example_images(images, title):
    # Select a random image from the specified subset to show to ensure that the subset is responsive
    print(title)
    random_index = np.random.randint(0, len(images))
    random_image = images[random_index]
    cv2.imshow(title, random_image)
    cv2.waitKey(0)


def train_base_architecture(base_model, image_generator, validation_generator=None):
    #  Freeze the layers of the base model architecture
    base_model.trainable = False

    # Add layers on top of base model to create functionality for specific prediction task
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(image_generator, epochs=3, validation_data=validation_generator)
    return model


def plot_cm(predicted, actual, model_name):
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    ConfusionMatrixDisplay.from_predictions(actual, predicted, ax=ax, normalize='true', cmap='inferno',
                                            xticks_rotation='vertical')
    plt.title(f"{model_name} Classification Confusion Matrix")
    plt.show()
