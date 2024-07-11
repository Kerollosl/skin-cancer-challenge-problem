"""This code was adapted from the following link: https://keras.io/guides/keras_tuner/getting_started/
It is part of a tutorial in setting up a keras hyperparameter tuner with the keras_tuner library. The
build_model function and all keras_tuner methods were adapted for the desired functionality of this
challenge problem
"""

from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner

def build_model(hp):

    # Choose hyperparameters from specified ranges below and train for 5 epochs to compare with other combinations
    model = keras.Sequential()
    model.add(keras.Input(shape=train_features_shape))

    # Conv layers and pooling layers
    for i in range(hp.Int("num_conv_layers", 1, 8)):
        model.add(layers.Conv2D(
            filters=hp.Int(f"filters_{i}", min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice(f"kernel_size_{i}", [3, 5]),
            activation='relu',
            padding='same'
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # Dense Layers
    for i in range(hp.Int("num_dense_layers", 1, 4)):
        model.add(layers.Dense(
            units=hp.Int(f"dense_units_{i}", min_value=32, max_value=512, step=64),
            activation='relu',
        ))
        if hp.Boolean("dropout_dense"):
            model.add(layers.Dropout(rate=hp.Float("dropout_rate_dense", min_value=0.1, max_value=0.5, step=0.1)))

    # Output Layer
    model.add(layers.Dense(1, activation="sigmoid"))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tune_hyperparameters(train_generator, validation_generator):
    # Global used to pass image shape to build_model function without breaking its hyperparameter tuning functionality
    global train_features_shape
    train_features_shape = validation_generator.image_shape

    model = build_model(keras_tuner.HyperParameters())
    print(model.summary())

    # Bayesian Optimization to efficiently select from hyperparameter combinations
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=build_model,
        objective="loss",
        max_trials=3,
        executions_per_trial=1,
        overwrite=True,
    )

    print(tuner.search_space_summary())

    tuner.search(validation_generator, epochs=5)
    models = tuner.get_best_models(num_models=1)
    best_model = models[0]
    print(best_model.summary())
    print(tuner.results_summary())

    best_hps = tuner.get_best_hyperparameters(1)
    # Build and train the model with the best hyperparameters.
    model = build_model(best_hps[0])
    model.fit(train_generator, epochs=3, validation_data=validation_generator)
    print(model.summary())
    return model
