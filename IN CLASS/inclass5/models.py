from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D ,Dropout
from keras.layers import BatchNormalization

class DeepANN:
    def train_model(self, model_instance, train_generator, validation_generator, epochs=5):
        mhistory = model_instance.fit(train_generator, validation_data=validation_generator, epochs=epochs)
        return mhistory

    def cnn_model_with_regularization(self, num_classes):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(29, 29, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))  # Add dropout with 20% probability

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))  # Adjust dropout rates for deeper layers

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Output layer for multi-class classification
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model