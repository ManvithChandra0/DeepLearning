import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

    def build_model(self, input_shape):
        # Encoder
        inputs = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = Flatten()(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # Reparameterization Trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

        # Decoder
        decoder_inputs = Input(shape=(self.latent_dim,))
        x = Dense(7 * 7 * 64, activation='relu')(decoder_inputs)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
        outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

        # Define models
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        decoder = Model(decoder_inputs, outputs, name='decoder')
        vae_outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, vae_outputs, name='vae')

        # Add KL Divergence loss
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae.add_loss(K.mean(kl_loss))

        return vae, encoder, decoder

