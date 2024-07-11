import pandas as pd
from keras_tuner_script import tune_hyperparameters
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3

from functions import *

CHOICES = ["Inception_V3", "VGG16", "tuned_model"]
MODEL_CHOICE = CHOICES[1]  # Select one of three architectures tested. Architecture of choice: VGG16
print(f"Model Choice: {MODEL_CHOICE}")

if __name__ == "__main__":
    train_data_dir = './skin_cancer_dataset/train'
    test_data_dir = './skin_cancer_dataset/test'

    # cv2 image tensors to be used for showing examples and later predicting for test images
    train_malignant_images = make_image_tensor('./skin_cancer_dataset/train/Malignant')
    train_benign_images = make_image_tensor('./skin_cancer_dataset/train/Benign')
    test_malignant_images = make_image_tensor('./skin_cancer_dataset/test/Malignant')
    test_benign_images = make_image_tensor('./skin_cancer_dataset/test/Benign')

    # Show example image from each subset within the main directory
    show_example_images(train_malignant_images, 'Train Malignant')
    show_example_images(train_benign_images, 'Train Benign')
    show_example_images(test_malignant_images, 'Test Malignant')
    show_example_images(test_benign_images, 'Test Benign')

    batch_size = 64

    """" ImageDataGenerator used for image transformation, resizing, and validation set splitting. 
         Final results were better without drastic transformation"""
    train_datagen = ImageDataGenerator(rescale=1./255,
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(75, 75),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(75, 75),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_data_dir,
        target_size=(75, 75),
        batch_size=batch_size,
        class_mode='binary',
    )

    # Select and train model of choice
    if MODEL_CHOICE == "Inception_V3":
        inception = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=train_generator.image_shape
        )
        model = train_base_architecture(inception, train_generator, validation_generator)

    elif MODEL_CHOICE == "VGG16":
        vgg = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=train_generator.image_shape
        )
        model = train_base_architecture(vgg, train_generator, validation_generator)

    else:
        model = tune_hyperparameters(train_generator, validation_generator)

    # Get predictions and append to df
    test_benign_image_names = os.listdir(test_data_dir+"/Benign")
    test_malignant_image_names = os.listdir(test_data_dir+"/Malignant")
    df = pd.DataFrame(columns=["Image_Name", "Prediction Value", "Classification", "Actual"])

    for index, image in enumerate(test_benign_images):
        image_name = test_benign_image_names[index]
        image = image * 1. / 255
        formatted_image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
        prediction = model.predict(formatted_image)[0][0]
        classification = "Malignant" if prediction > 0.5 else "Benign"
        df.loc[index] = [image_name, prediction, classification, "Benign"]

    for index, image in enumerate(test_malignant_images):
        image_name = test_malignant_image_names[index]
        image = image * 1./255
        formatted_image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
        prediction = model.predict(formatted_image)[0][0]
        classification = "Malignant" if prediction > 0.5 else "Benign"
        df.loc[index+1000] = [image_name, prediction, classification, "Malignant"]

    print(df.head())


    # Calculate test set accuracy
    df["Correct Prediction"] = df["Classification"] == df["Actual"]
    value_counts = df["Correct Prediction"].value_counts()
    correct_predictions = value_counts.iloc[0]
    incorrect_predictions = value_counts.iloc[1]
    accuracy = correct_predictions / (correct_predictions + incorrect_predictions)*100
    print(f"Accuracy Score: {accuracy}%")

    # Plot confusion matrix, save preds to csv, and save model
    plot_cm(df["Classification"], df["Actual"], MODEL_CHOICE)
    df.to_csv("generated_preds.csv", index=False)
    model.save("model.h5")
