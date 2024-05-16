from models import VariationalAutoencoder
from preprocessing import ImageDataLoader
import matplotlib.pyplot as plt
import tensorflow as tf

def display_images(original_images, reconstructed_images, num_images=5):
    fig, axs = plt.subplots(2, num_images, figsize=(10, 5))
    for i in range(num_images):
        axs[0, i].imshow(original_images[i])
        axs[0, i].set_title('Original')
        axs[0, i].axis('off')
        axs[1, i].imshow(reconstructed_images[i])
        axs[1, i].set_title('Reconstructed')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.show()


def variational_autoencoder_call():
    # Create an instance of the VariationalAutoencoder class
    vae = VariationalAutoencoder(latent_dim=2)

    # Define the directory containing the data (train and test images)
    data_directory = "/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw"

    # Obtain the variational autoencoder model and data generator
    vae_model, encoder, decoder = vae.build_model(input_shape=(28, 28, 3))
    data_loader = ImageDataLoader(data_directory)
    data_flow = data_loader.load_data()

    # Compile the VAE model
    vae_model.compile(optimizer='adam', loss='binary_crossentropy')

    # Train the variational autoencoder model using the data generator
    history = vae_model.fit(data_flow, epochs=10, validation_data=data_flow)

    # Evaluate the model on the data generator
    loss = vae_model.evaluate(data_flow)
    print("Test loss:", loss)

    # Load a batch of images from the data generator
    test_images, _ = next(data_flow)

    # Reconstruct images using the variational autoencoder
    reconstructed_images = vae_model.predict(test_images)

    # Display original and reconstructed images side by side
    display_images(test_images, reconstructed_images, 10)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()



def main():
    variational_autoencoder_call()


if __name__ == "__main__":
    main()