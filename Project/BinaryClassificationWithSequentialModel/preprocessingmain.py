import preprocessing as pp
from models import DeepANN
import matplotlib.pyplot as plt
import json
import os

if __name__ == '__main__':
    '''    
    data = pp.PreprocessData()
    image_df, train, label = data.preprocess("/Users/manvithchandra/Downloads/DL data set")

    tr_gen, tt_gen, va_gen = data.generate_train_test_images(image_df)

    ann_model = DeepANN()

    # Simple Model
    model_simple = ann_model.simple_model()
    print("train_generator", tr_gen)

    # Fit the model
    history = model_simple.fit(tr_gen, validation_data=va_gen, epochs=10)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training and Validation Accuracy: BinaryClassificationWithSequentialModel')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
'''
    def plot_history(history):
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


    def main():
        json_file_path = 'model_history.json'

        if os.path.exists(json_file_path):
            print("Loading model history from JSON file.")
            with open(json_file_path, 'r') as file:
                history = json.load(file)
            plot_history(history)
        else:
            print("JSON file not found. Initiating data preprocessing, model training, and saving new history...")
            data = pp.PreprocessData()
            image_df, train, label = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")  # Change this to your dataset path

            tr_gen, tt_gen, va_gen = data.generate_train_test_images(image_df)

            ann_model = DeepANN()
            model_simple = ann_model.simple_model()

            history = model_simple.fit(tr_gen, validation_data=va_gen, epochs=10).history

            # Save the new model history
            with open(json_file_path, 'w') as file:
                json.dump(history, file)

            plot_history(history)


    if __name__ == '__main__':
        main()