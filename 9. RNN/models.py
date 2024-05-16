from keras.layers import LSTM
from tensorflow.keras.layers import Dense ,Reshape
from keras.models import Sequential


class DeepRNN:
    def rnn_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Reshape((input_shape[0], input_shape[1] * input_shape[2]), input_shape=input_shape))
        model.add(LSTM(128))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
        return model
