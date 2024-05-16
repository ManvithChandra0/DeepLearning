import preprocessing as pp
from models import DeepCNNLSTM
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    df, _, _ = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")

    train_gen, test_gen, validate_gen = data.generate_train_test_images(df)

    cnn_lstm_model = DeepCNNLSTM()  # Use the modified class with LSTM

    # CNN + LSTM Model
    model_cnn_lstm = cnn_lstm_model.cnn_lstm_model()

    # Fit the model
    history = model_cnn_lstm.fit(
        train_gen,
        validation_data=validate_gen,
        epochs=10,
        verbose=1
    )

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('10.CNNLSTM Model Training and Validation Accuracy with Data Augmentation and Regularization')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

