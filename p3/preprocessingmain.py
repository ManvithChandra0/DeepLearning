import preprocessing as pp
from models import DeepANN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    df, _, _ = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")

    train_gen, test_gen, validate_gen = data.generate_train_test_images(df)

    ann_model = DeepANN()

    model = ann_model.simple_model(num_classes=len(train_gen.class_indices))

    history = model.fit(train_gen, validation_data=validate_gen, epochs=10, verbose=1)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training and Validation Accuracy with Sequential model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
