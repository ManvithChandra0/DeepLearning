from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential

class DeepANN:
    def __init__(self):
        pass

    def cnn_model_with_regularization(self, num_classes):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model