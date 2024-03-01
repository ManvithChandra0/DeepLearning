from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2

class DeepCNN:
    def cnn_model(self, input_shape=(29, 29, 3), num_classes=2, dropout_rate=0.5):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
