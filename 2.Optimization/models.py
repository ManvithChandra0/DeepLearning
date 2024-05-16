from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

class DeepANN:
    def simple_model(self, optimizer="sgd"):
        model = Sequential()
        model.add(Flatten(input_shape=(29, 29, 3)))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))  # Assuming you have 2 classes for LFW

        if optimizer == "sgd":
            opt = SGD()
        elif optimizer == "adam":
            opt = Adam()
        elif optimizer == "rmsprop":
            opt = RMSprop()
        else:
            raise ValueError("Invalid optimizer specified")

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    # sgd,adam,rms
