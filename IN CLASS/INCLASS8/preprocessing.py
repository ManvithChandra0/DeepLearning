import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class PreprocessData:
    def __init__(self):
        self.num_classes = 0

    def preprocess(self, dir_path):
        train = []
        labels = []

        label_encoder = LabelEncoder()

        for class_label in os.listdir(dir_path):
            class_path = os.path.join(dir_path, class_label)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    train.append(img_path)
                    labels.append(class_label)

        unique_classes = sorted(set(labels))
        num_classes = len(unique_classes)

        print(f"Number of classes: {num_classes}")
        print("Unique classes:", unique_classes)

        encoded_labels = label_encoder.fit_transform(labels)

        return train, encoded_labels, num_classes

    def generate_train_test_images(self, train, encoded_labels, test_size=0.2, random_state=42):
        train_data, test_data, train_labels, test_labels = train_test_split(train, encoded_labels, test_size=test_size, random_state=random_state)

        train_df = pd.DataFrame({'Image': train_data, 'Labels': train_labels})
        test_df = pd.DataFrame({'Image': test_data, 'Labels': test_labels})

        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='Image', y_col='Labels',
            target_size=(150, 150), batch_size=32, class_mode='raw')

        test_generator = test_datagen.flow_from_dataframe(
            test_df,
            x_col='Image', y_col='Labels',
            target_size=(150, 150), batch_size=32, class_mode='raw')

        return train_generator, test_generator
