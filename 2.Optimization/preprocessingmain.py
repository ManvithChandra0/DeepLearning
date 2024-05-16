import preprocessing as pp
import models as ms
import matplotlib.pyplot as plt

if __name__ == '__main__':
    obj = pp.preprocess_data()
    dir_path = "D:\DL SKILLWORK\PROJECT EXP2(VISUALIZING PRO DATASET)\EuroSAT"  # Update with the correct path
    weed_df, train, labels = obj.preprocess(dir_path)

    tr_gen, tt_gen, va_gen = obj.generate_train_test_images(weed_df, train, labels)

    ANN_model = ms.DeepANN()

    # Create models with different optimizers
    model_sgd = ANN_model.simple_model(optimizer="sgd")
    model_adam = ANN_model.simple_model(optimizer="adam")
    model_rmsprop = ANN_model.simple_model(optimizer="rmsprop")

    # Train models
    ann_history_sgd = model_sgd.fit(tr_gen, epochs=10, validation_data=va_gen)
    ann_history_adam = model_adam.fit(tr_gen, epochs=10, validation_data=va_gen)
    ann_history_rmsprop = model_rmsprop.fit(tr_gen, epochs=10, validation_data=va_gen)

    # Evaluate models
    ann_test_loss_sgd, ann_test_acc_sgd = model_sgd.evaluate(tt_gen)
    ann_test_loss_adam, ann_test_acc_adam = model_adam.evaluate(tt_gen)
    ann_test_loss_rmsprop, ann_test_acc_rmsprop = model_rmsprop.evaluate(tt_gen)

    print(f"Test Accuracy (SGD): {ann_test_acc_sgd}")
    print(f"Test Accuracy (Adam): {ann_test_acc_adam}")
    print(f"Test Accuracy (RMSprop): {ann_test_acc_rmsprop}")

    # Save models
    model_sgd.save("mymodel_sgd.keras")
    model_adam.save("mymodel_adam.keras")
    model_rmsprop.save("mymodel_rmsprop.keras")

    print("SGD Model Architecture:")
    print(model_sgd.summary())

    print("Adam Model Architecture:")
    print(model_adam.summary())

    print("RMSprop Model Architecture:")
    print(model_rmsprop.summary())

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.plot(ann_history_sgd.history['loss'], label='Train loss')
    plt.plot(ann_history_sgd.history['val_loss'], label='Validation loss')
    plt.title('SGD Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(ann_history_adam.history['loss'], label='Train loss')
    plt.plot(ann_history_adam.history['val_loss'], label='Validation loss')
    plt.title('Adam Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(ann_history_rmsprop.history['loss'], label='Train loss')
    plt.plot(ann_history_rmsprop.history['val_loss'], label='Validation loss')
    plt.title('RMSprop Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
