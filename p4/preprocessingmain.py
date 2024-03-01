import preprocessing as pp
from models import DeepANN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pp.PreprocessData()
    df, _, _ = data.preprocess("/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw")

    train_gen, test_gen, validate_gen = data.generate_train_test_images(df)

    ann_model = DeepANN()

    # Simple Model
    model = ann_model.simple_model(num_classes=len(train_gen.class_indices))

    # Fit the model using fit_generator for random mini-batch evaluations
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=10,
        validation_data=validate_gen,
        validation_steps=len(validate_gen),
        verbose=1
    )

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training and Validation Accuracy : Mini-batch evaluations')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
