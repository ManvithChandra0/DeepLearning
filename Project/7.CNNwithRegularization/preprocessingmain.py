import preprocessing as pp
from models import DeepCNN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    df, _, _ = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")

    train_gen, test_gen, validate_gen = data.generate_train_test_images(df)

    cnn_model = DeepCNN()

    # CNN Model
    model_cnn = cnn_model.cnn_model()

    # Fit the model
    history = model_cnn.fit(
        train_gen,
        validation_data=validate_gen,
        epochs=10,
        verbose=1
    )

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Model Training and Validation Accuracy with Data Augmentation and Regularization')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
