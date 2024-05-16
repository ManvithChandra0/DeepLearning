import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model


class DenoisingAutoencoder(tf.keras.Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

    def build_model(self):
        input_img = Input(shape=(28, 28, 3))

        # Add Gaussian noise to the input images
        noise = tf.keras.layers.GaussianNoise(0.2)(input_img)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(noise)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        return autoencoder
