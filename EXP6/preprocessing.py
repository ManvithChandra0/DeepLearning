import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
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

        # Print unique classes
        unique_classes = sorted(set(labels))
        num_classes = len(unique_classes)
        print(f"Number of classes: {num_classes}")
        print("Unique classes:", unique_classes)

        encoded_labels = label_encoder.fit_transform(labels)

        return train, encoded_labels, num_classes

    def generate_train_test_images(self, train, encoded_labels):
        train_df = pd.DataFrame({'Image': train, 'Labels': encoded_labels})

        train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=42)

        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            train_data,
            x_col='Image',
            y_col='Labels',
            target_size=(150, 150),
            batch_size=32,
            class_mode='raw'
        )

        test_generator = test_datagen.flow_from_dataframe(
            test_data,
            x_col='Image',
            y_col='Labels',
            target_size=(150, 150),
            batch_size=32,
            class_mode='raw'
        )

        return train_generator, test_generator