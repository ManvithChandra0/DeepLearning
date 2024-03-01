import preprocessing as pp
from models import VGGModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    df, _, _ = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")

    train_gen, test_gen, validate_gen = data.generate_train_test_images(df)

    vgg_model = VGGModel()

    # VGG Model
    model_vgg = vgg_model.vgg_model()

    # Fit the model
    history = model_vgg.fit(train_gen, validation_data=validate_gen, epochs=10, verbose=1)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('VGG Model Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
