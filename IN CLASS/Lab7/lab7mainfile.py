import numpy as np
import matplotlib.pyplot as plt
from lab7classfile import AdamOptimizerModel

# Generate some random data for demonstration
np.random.seed(42)
X_train = np.random.rand(100, 1)
y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1)

# Create an instance of the AdamOptimizerModel class
model_instance = AdamOptimizerModel()

# Compile the model
model_instance.compile_model()

# Train the model in batches of 10 epochs, up to a total of 100 epochs
history = model_instance.train_model(X_train, y_train, total_epochs=100, batch_size=10)

# Evaluate the model on the training data
loss = model_instance.evaluate_model(X_train, y_train)
print("Final training loss:", loss)

# Analyze the training history
plt.plot(history['loss'])
plt.title('Model Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
