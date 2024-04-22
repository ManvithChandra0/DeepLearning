from models import DenoisingAutoencoder
from preprocessing import ImageDataLoader
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def display_images(original_images, noisy_images, denoised_images, num_images=5):
    fig, axs = plt.subplots(3, num_images, figsize=(10, 10))
    for i in range(num_images):
        # Original image
        axs[0, i].imshow(original_images[i])
        axs[0, i].set_title('Original')
        axs[0, i].axis('off')
        # Noisy image
        axs[1, i].imshow(noisy_images[i])
        axs[1, i].set_title('Noisy')
        axs[1, i].axis('off')
        # Denoised image
        axs[2, i].imshow(denoised_images[i])
        axs[2, i].set_title('Denoised')
        axs[2, i].axis('off')
    plt.tight_layout()
    plt.show(block=True)


def denoising_autoencoder_call():
    # Create an instance of the 12.DenoisingAutoencoder class
    denoising_autoen = DenoisingAutoencoder()

    # Define the directory containing the data (train and test images)
    data_directory = "/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw"

    # Obtain the denoising autoencoder model and data generator
    autoencoder_model = denoising_autoen.build_model()
    data_loader = ImageDataLoader(data_directory)
    data_flow = data_loader.load_data()

    # Train the denoising autoencoder model using the data generator
    history = autoencoder_model.fit(data_flow, epochs=10, validation_data=data_flow)

    # Save the trained denoising autoencoder model
    autoencoder_model.save('denoising_autoencoder_saved.h5')

    # Evaluate the model on the data generator
    loss = autoencoder_model.evaluate(data_flow)
    print("Test loss:", loss)

    # Load the saved denoising autoencoder model
    model = tf.keras.models.load_model('denoising_autoencoder_saved.h5')

    # Load a batch of images from the data generator
    test_images, _ = next(data_flow)

    # Add noise to the test images
    noisy_test_images = test_images + np.random.normal(loc=0.0, scale=0.2, size=test_images.shape)
    noisy_test_images = np.clip(noisy_test_images, 0., 1.)

    # Denoise the noisy images using the autoencoder
    denoised_images = autoencoder_model.predict(noisy_test_images)

    # Display original, noisy, and denoised images side by side
    display_images(test_images, noisy_test_images, denoised_images, 10)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def main():
    denoising_autoencoder_call()


if __name__ == "__main__":
    main()