import os
import cv2
import pandas as pd
from matplotlib import pyplot as plt

class ImageProcessor:
    def __init__(self):
        self.data = {'Image': [], 'Label': []}

    def load_images(self, dir_path):
        for category in os.listdir(dir_path):
            category_path = os.path.join(dir_path, category)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    file_path = os.path.join(category_path, file)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data['Image'].append(file_path)
                        self.data['Label'].append(category)

    def visualize_images(self, nimages):
        num_categories = len(set(self.data['Label']))
        fig, axs = plt.subplots(num_categories, nimages, figsize=(10, 10))

        for i, category in enumerate(set(self.data['Label'])):
            category_images = [img for img, label in zip(self.data['Image'], self.data['Label']) if label == category]
            for j in range(min(nimages, len(category_images))):
                img_path = category_images[j]
                img = cv2.imread(img_path)

                axs[i][j].title.set_text(category)
                axs[i][j].imshow(img)

        fig.tight_layout()
        plt.show(block=True)

    def create_data_file(self, output_file):
        df = pd.DataFrame(self.data)
        df.to_csv(output_file, index=False)
        print(f"Data file '{output_file}' created successfully.")

# Example usage:
dir_path = "../gaussian_filtered_images"
output_file = "../image_data.csv"

processor = ImageProcessor()

processor.load_images(dir_path)

processor.visualize_images(nimages=2)

processor.create_data_file(output_file)
