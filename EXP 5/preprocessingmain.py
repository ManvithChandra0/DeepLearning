# plotmain.py
import preprocessing as pp
from models import DeepANN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    image_df, train, label = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")
    image_df.to_csv('lfw.csv')

    tr_gen, tt_gen, va_gen = data.generate_train_test_images(image_df)

    ann_model = DeepANN()

    # Simple Model
    model_simple = ann_model.simple_model()
    print("train_generator", tr_gen)

    # CNN Model
    model_cnn = ann_model.cnn_model()
    cnn_history = ann_model.train_model(model_cnn, tr_gen, va_gen, epochs=10)

    plt.plot(cnn_history.history['accuracy'], label='CNN Model')
    plt.title('Model Training accuracy comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=True)
