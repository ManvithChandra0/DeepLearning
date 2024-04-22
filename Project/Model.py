from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

class Model:
    def __init__(self):
        pass

    def simple_ANN(self, input_shape, optimizer):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model



    def simple_model(self, input_shape=(29, 29, 3), optimizer="sgd"):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))  # Binary classification requires a single neuron with sigmoid activation
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    def optimization(self, optimizer="sgd"):
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
