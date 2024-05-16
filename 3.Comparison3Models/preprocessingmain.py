import preprocessing as pp
from models import DeepANN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    df, _, _ = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")

    train_gen, test_gen, validate_gen = data.generate_train_test_images(df)

    ann_model = DeepANN()

    # SGD optimizer
    model_sgd = ann_model.simple_model(optimizer="sgd")
    history_sgd = model_sgd.fit(train_gen, validation_data=validate_gen, epochs=10, verbose=1)

    # Adam optimizer
    model_adam = ann_model.simple_model(optimizer="adam")
    history_adam = model_adam.fit(train_gen, validation_data=validate_gen, epochs=10, verbose=1)

    # RMSprop optimizer
    model_rmsprop = ann_model.simple_model(optimizer="rmsprop")
    history_rmsprop = model_rmsprop.fit(train_gen, validation_data=validate_gen, epochs=10, verbose=1)

    plt.plot(history_sgd.history['accuracy'], label='SGD')
    plt.plot(history_adam.history['accuracy'], label='Adam')
    plt.plot(history_rmsprop.history['accuracy'], label='RMSprop')
    plt.title('Model Training Accuracy Comparison with Different Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
