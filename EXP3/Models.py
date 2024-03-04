from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

class DeepANN:
    def simple_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(29, 29, 3)))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))  # Assuming you have 2 classes for LFW
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model
