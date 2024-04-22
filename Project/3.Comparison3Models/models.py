from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

class DeepANN:
    def simple_model(self, input_shape=(29, 29, 3), optimizer="sgd"):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))  # Binary classification requires a single neuron with sigmoid activation

        if optimizer == "sgd":
            opt = SGD(learning_rate=0.01)
        elif optimizer == "adam":
            opt = Adam(learning_rate=0.001)
        elif optimizer == "rmsprop":
            opt = RMSprop(learning_rate=0.001)
        else:
            raise ValueError("Invalid optimizer specified.")

        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
