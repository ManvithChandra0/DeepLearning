import tensorflow as tf
from tensorflow.keras import layers

class AdamOptimizerModel:
    def __init__(self):
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(units=1)
        ])
        return model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')

    def train_model(self, X_train, y_train, total_epochs=100, batch_size=10):
        history = {'loss': []}
        for epoch in range(0, total_epochs, batch_size):
            current_epochs = min(batch_size, total_epochs - epoch)
            batch_history = self.model.fit(X_train, y_train, epochs=current_epochs, verbose=0)
            history['loss'].extend(batch_history.history['loss'])
        return history

    def evaluate_model(self, X_test, y_test):
        loss = self.model.evaluate(X_test, y_test)
        return loss
