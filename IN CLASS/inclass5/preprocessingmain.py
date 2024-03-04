import preprocessing as pp
from models import DeepANN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    image_df, train, encoded_labels = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/gaussian_filtered_images")

    tr_gen, tt_gen, va_gen = data.generate_train_test_images(image_df)

    # Dynamically determine the number of classes
    num_classes = len(set(encoded_labels.argmax(axis=1)))

    print("Number of classes:", num_classes)
    print("Shape of one-hot encoded labels:", one_hot_labels.shape)

    ann_model = DeepANN()
    model_cnn_with_reg = ann_model.cnn_model_with_regularization(num_classes=num_classes)

    # Print the summary of the model
    model_cnn_with_reg.summary()

    cnn_history_with_reg = ann_model.train_model(model_cnn_with_reg, tr_gen, va_gen, epochs=10)
    plt.plot(cnn_history_with_reg.history['accuracy'], label='CNN Model with Batch Normalization')
    plt.title('Model Training accuracy comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=True)