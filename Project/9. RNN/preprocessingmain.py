import preprocessing as pp
import models as m
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    train, encoded_labels, num_classes = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")
    train_generator, test_generator = data.generate_train_test_images(train, encoded_labels)

    input_shape = (150, 150, 3)

    rnn_model = m.DeepRNN()
    model_rnn = rnn_model.rnn_model(input_shape=input_shape, num_classes=num_classes)
    model_rnn.summary()

    rnn_history = model_rnn.fit(train_generator,
                                validation_data=test_generator,
                                epochs=10)

    plt.plot(rnn_history.history['accuracy'], label='Training Accuracy')
    plt.plot(rnn_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
