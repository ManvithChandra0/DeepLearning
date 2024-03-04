from models import Autoencoder
from preprocessing import ImageDataLoader
import matplotlib.pyplot as plt
import tensorflow as tf

def display_images(original_images, predicted_images, num_images=5):
    fig, axs = plt.subplots(2, num_images, figsize=(10, 10))
    for i in range(num_images):
        # Original image
        axs[0, i].imshow(original_images[i])
        axs[0, i].set_title('Original')
        axs[0, i].axis('off')
        # Predicted image
        axs[1, i].imshow(predicted_images[i])
        axs[1, i].set_title('Predicted')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.show(block=True)


def autoencoder_call():
    # Create an instance of the Autoencoder class
    autoen = Autoencoder()

    # Define the directory containing the data (train and test images)
    data_directory = '/Users/manvithchandra/Desktop/DeepLearning/gaussian_filtered_images'

    # Obtain the autoencoder model and data generator
    autoencoder_model = autoen.build_model()
    data_loader = ImageDataLoader(data_directory)
    data_flow = data_loader.load_data()

    # Train the autoencoder model using the data generator
    history = autoencoder_model.fit(data_flow, epochs=50, validation_data=data_flow)

    # Save the trained autoencoder model
    autoencoder_model.save('autoencoder_saved.h5')

    # Evaluate the model on the data generator
    loss = autoencoder_model.evaluate(data_flow)
    print("Test loss:", loss)

    # Load the saved autoencoder model
    model = tf.keras.models.load_model('autoencoder_saved.h5')

    # Load a batch of images from the data generator
    test_images, _ = next(data_flow)

    # Predict images using the autoencoder
    predicted_images = autoencoder_model.predict(test_images)

    # Display original and predicted images side by side
    display_images(test_images, predicted_images, 10)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def main():
    autoencoder_call()


if __name__ == "__main__":
    main()
