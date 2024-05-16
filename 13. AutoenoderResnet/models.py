import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Add
from tensorflow.keras.models import Model


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ResNetBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.conv2 = Conv2D(filters, kernel_size, activation=None, padding='same')
        self.conv3 = Conv2D(filters, (1, 1), activation=None, padding='same')  # Additional Conv2D layer

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        if inputs.shape[-1] != x.shape[-1]:
            inputs = self.conv3(inputs)  # Using the additional Conv2D layer

        x = Add()([inputs, x])
        x = tf.nn.relu(x)
        return x


class AutoencoderResNet(tf.keras.Model):
    def __init__(self):
        super(AutoencoderResNet, self).__init__()

    def build_model(self):
        input_img = Input(shape=(28, 28, 3))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = ResNetBlock(32, (3, 3))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = ResNetBlock(32, (3, 3))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        return autoencoder


