import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
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

        # Encode labels to categorical format
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        one_hot_labels = to_categorical(encoded_labels, num_classes=num_classes)

        print("Number of images: {}\n".format(len(train)))
        print("Number of image labels: {}\n".format(len(one_hot_labels)))
        df = pd.DataFrame({'Image': train, 'Labels': one_hot_labels.tolist()})
        print(df)
        return df, train, one_hot_labels

    def generate_train_test_images(self, df):
        train, test = train_test_split(df, test_size=0.2)
        print("Train:")
        print(train)

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.15
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            train,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(29, 29),
            color_mode="rgb",
            class_mode="categorical",
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
            class_mode="categorical",
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
            class_mode="categorical",
            batch_size=10
        )

        print(f"Train images shape: {train.shape}")
        print(f"Testing images shape: {test.shape}")
        return train_generator, test_generator, validate_generator
