import Preprocessing as pp
from Models import DeepANN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.preprocess_data()
    image_df, train, label = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")
    image_df.to_csv('lfw.csv')

    tr_gen, tt_gen, va_gen = data.generate_train_test_images(image_df, train, label)

    # Create and train the simple neural network model
    ann_model = DeepANN()
    model_simple = ann_model.simple_model()
    print("train_generator", tr_gen)
    # Compare simple models
    image_shape = (28, 28, 3)
    model_adam = ann_model.simple_model(input_shape=image_shape, optimizer='adam')
    model_sgd = ann_model.simple_model(input_shape=image_shape, optimizer='sgd')
    model_rmsprop = ann_model.simple_model(input_shape=image_shape, optimizer='rmsprop')

    ann_model.compare_models([model_adam, model_sgd, model_rmsprop], tr_gen, va_gen, epochs=3)
    
    
    

    '''
    # Create and train the CNN model
    model_cnn = ann_model.cnn_model()
    cnn_history = ann_model.train_model(model_cnn, tr_gen, va_gen, epochs=3)

    # Plot CNN model accuracy
    plt.plot(cnn_history.history['accuracy'], label='CNN Model')
    plt.title('Model Training accuracy comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=True)
    
    '''