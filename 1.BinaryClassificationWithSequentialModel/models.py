from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

class DeepANN:
    def simple_model(self, input_shape=(29, 29, 3), optimizer="sgd"):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))  # Binary classification requires a single neuron with sigmoid activation
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model
