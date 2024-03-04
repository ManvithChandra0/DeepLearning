from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Reshape, Dense, Dropout


class DeepANN:


    def create_lstm_model(input_shape, num_classes):
        model = Sequential()
        # Reshape layer to flatten the input images
        model.add(Reshape((input_shape[0], input_shape[1] * input_shape[2]), input_shape=input_shape))
        # LSTM layer
        model.add(LSTM(128))  # Adjust units as needed
        # Fully connected layers
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

