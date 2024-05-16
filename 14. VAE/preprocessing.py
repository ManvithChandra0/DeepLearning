from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageDataLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self):
        data_generator = ImageDataGenerator(rescale=1. / 255)

        data_flow = data_generator.flow_from_directory(
            self.directory,
            target_size=(28, 28),
            batch_size=32,
            class_mode='input'
        )

        return data_flow
