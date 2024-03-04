# preprocessingmain.py

import preprocessing as pp
from models import DeepANN
import matplotlib.pyplot as plt





if __name__ == '__main__':
    data = pp.PreprocessData()
    train, encoded_labels, num_classes = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/gaussian_filtered_images")
    train_generator, test_generator = data.generate_train_test_images(train, encoded_labels)

    ann_model = DeepANN()
    model_cnn_with_reg = ann_model.cnn_model_with_regularization(num_classes=num_classes)
    model_cnn_with_reg.summary()

    cnn_history_with_reg = model_cnn_with_reg.fit(train_generator, validation_data=test_generator, epochs=10)

    plt.plot(cnn_history_with_reg.history['accuracy'], label='Training Accuracy')
    plt.plot(cnn_history_with_reg.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()