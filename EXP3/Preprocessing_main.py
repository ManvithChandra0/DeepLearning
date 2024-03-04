from EXP.EXP3 import Preprocessing as pp
import Models as ms
import matplotlib.pyplot as plt

obj = pp.preprocess_data()
dir_path = "/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw"  # Update with the correct path
weed_df, train, labels = obj.preprocess(dir_path)

tr_gen, tt_gen, va_gen = obj.generate_train_test_images(weed_df, train, labels)

ANN_model = ms.DeepANN()
model1 = ANN_model.simple_model()
print("train generator", tr_gen)
ann_history = model1.fit(tr_gen, epochs=100, validation_data=va_gen)

ann_test_loss, ann_test_acc = model1.evaluate(tt_gen)
print(f"Test Accuracy: {ann_test_acc}")


print("the ann Architecture")
print(model1.summary())

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(ann_history.history['loss'], label='Train loss')
plt.plot(ann_history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ann_history.history['accuracy'], label='Train accuracy')
plt.plot(ann_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()