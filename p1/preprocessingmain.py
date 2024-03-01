import preprocessing as pp
from models import DeepANN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    image_df, train, label = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")

    tr_gen, tt_gen, va_gen = data.generate_train_test_images(image_df)

    ann_model = DeepANN()

    # Simple Model
    model_simple = ann_model.simple_model()
    print("train_generator", tr_gen)

    # Fit the model
    history = model_simple.fit(tr_gen, validation_data=va_gen, epochs=10)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training and Validation Accuracy: Binary Classification with Sequential Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
