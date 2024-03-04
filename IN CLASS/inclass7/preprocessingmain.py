import preprocessing as pp
import numpy as np
from models import DeepANN
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

if __name__ == '__main__':
    # Instantiate PreprocessData class
    data_processor = pp.PreprocessData()

    # Preprocess the data
    images, encoded_labels, num_classes = data_processor.preprocess(
        "/Users/manvithchandra/Desktop/DeepLearning/gaussian_filtered_images")

    # Load and preprocess images
    X_images = []
    for img_path in images:
        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        X_images.append(img_array)

    # Convert to numpy array
    X_images = np.array(X_images)

    # Reshape images to match the expected input shape of the LSTM layer
    X_images = X_images.reshape(X_images.shape[0], -1, 150 * 150 * 3)

    # Generate train and test data splits
    X_train, X_test, y_train, y_test = data_processor.generate_train_test_images(X_images, encoded_labels)

    # Instantiate DeepANN class
    ann_model = DeepANN()

    # Build the RNN model
    model_rnn_with_reg = ann_model.rnn_model_with_regularization(num_classes=num_classes)
    model_rnn_with_reg.summary()

    # Train the RNN model
    rnn_history_with_reg = model_rnn_with_reg.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    # Evaluate the trained model
    test_loss, test_accuracy = model_rnn_with_reg.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy}')

    # Plot training/validation accuracy
    plt.plot(rnn_history_with_reg.history['accuracy'], label='Training Accuracy')
    plt.plot(rnn_history_with_reg.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
