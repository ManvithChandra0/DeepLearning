from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.regularizers import l2


from tensorflow.keras.layers import Reshape

class DeepCNNLSTM:
    def cnn_lstm_model(self, input_shape=(29, 29, 3), num_classes=2, dropout_rate=0.5):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # Reshape to add time dimension for LSTM
        model.add(Reshape((1, -1)))
        # Add LSTM layer
        model.add(LSTM(64))  # You can adjust the number of units as needed
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
