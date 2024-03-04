import matplotlib.pyplot as plt
from preprocessing import PreprocessData
from models import create_lstm_model


if __name__ == '__main__':
    data = PreprocessData()
    train, encoded_labels, num_classes = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")
    train_generator, test_generator = data.generate_train_test_images(train, encoded_labels, test_size=0.2, random_state=42)

    input_shape = (150, 150, 3)

    lstm_model = create_lstm_model(input_shape=input_shape, num_classes=num_classes)
    lstm_model.summary()

    lstm_history = lstm_model.fit(train_generator,
                                  validation_data=test_generator,
                                  epochs=10)

    plt.plot(lstm_history.history['accuracy'], label='Training Accuracy')
    plt.plot(lstm_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()