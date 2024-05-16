import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class PreprocessData:
    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        labels = []

        for i in os.listdir(dpath):
            if not i.startswith('.'):
                path_to_check = os.path.join(dpath, i)

                if os.path.isdir(path_to_check):
                    train_class = os.listdir(path_to_check)
                    for j in train_class:
                        img = os.path.join(path_to_check, j)
                        train.append(img)
                        labels.append(i)

        # Assuming binary classification
        binary_labels = [1 if label == 'positive' else 0 for label in labels]

        print("Number of images: {}\n".format(len(train)))
        print("Number of image labels: {}\n".format(len(binary_labels)))
        df = pd.DataFrame({'Image': train, 'Labels': binary_labels})
        print(df)
        return df, train, binary_labels

    def generate_train_test_images(self, df):
        train, test = train_test_split(df, test_size=0.2)
        print("Train:")
        print(train)

        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, validation_split=0.15)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            train,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(29, 29),
            color_mode="rgb",
            class_mode="raw",
            batch_size=10,
            subset='training'
        )

        validate_generator = train_datagen.flow_from_dataframe(
            train,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(29, 29),
            color_mode="rgb",
            class_mode="raw",
            batch_size=10,
            subset='validation'
        )

        test_generator = test_datagen.flow_from_dataframe(
            test,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(29, 29),
            color_mode="rgb",
            class_mode="raw",
            batch_size=10
        )

        print(f"Train images shape: {train.shape}")
        print(f"Testing images shape: {test.shape}")
        return train_generator, test_generator, validate_generator
