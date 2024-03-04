import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv2D, MaxPooling2D


class DeepANN:
    # def simple_model(self):
    #     model = Sequential()
    #     model.add(Flatten(input_shape=(28, 28, 3)))
    #     model.add(Dense(128, activation="relu"))
    #     model.add(Dense(64, activation="relu"))
    #     model.add(Dense(2, activation="softmax"))  # Assuming you have 2 classes for LFW
    #     model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    #     return model

    # sgd,adam,rms

    def simple_model(self,input_shape=(28, 28, 3),optimizer="sgd"):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))  # Assuming you have 2 classes for LFW
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model

    def train_model(self, model_instance, train_generator, validation_generator, epochs=5):
        mhistory = model_instance.fit(train_generator, validation_data=validation_generator, epochs=epochs)
        return mhistory

    def compare_models(self, models, train_generator, validate_generator, epochs=5):
        histories = []
        plt.figure(figsize=(10, 6))
        for i, model in enumerate(models):
            history = self.train_model(model, train_generator, validate_generator, epochs=epochs)
            histories.append(history)
            plt.plot(history.history['accuracy'], label=f'Model {i + 1}')

        plt.title('Model Training accuracy comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show(block=True)

    def cnn_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))  # 26 x 26
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(2, activation='softmax'))

        # Compile the model
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model

