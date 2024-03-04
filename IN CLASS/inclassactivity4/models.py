# models.py

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os

class DeepANN:
    def __init__(self):
        self.histories = []

    def simple_model(self, input_shape=(29, 29, 3), optimizer="sgd"):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))  # Assuming you have 2 classes for your problem
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    def train_model(self, model_instance, train_generator, validation_generator, epochs=5):
        history = model_instance.fit(train_generator, validation_data=validation_generator, epochs=epochs)
        self.histories.append(history)
        return history

    def compare_models(self, models, train_generator, validate_generator, epochs=5):
        self.histories = []

        # Create a directory to save comparison plots
        os.makedirs('static/plots', exist_ok=True)

        for i, model in enumerate(models):
            history = self.train_model(model, train_generator, validate_generator, epochs=epochs)
            plt.plot(history.history['accuracy'], label=f'Model {i + 1}')

        plt.title('Model Training Accuracy Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save the plot as an image file
        plot_filename = f'static/plots/comparison_plot.png'
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to avoid displaying it in the notebook

        return plot_filename

    def save_comparison_graph(self, html_path, plot_filename):
        with open(html_path, 'w') as html_file:
            html_file.write(f'''
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body {{
                            font-family: 'Arial', sans-serif;
                            background-color: #57c6f2; /* Light blue background color */
                            margin: 0;
                            padding: 0;
                            text-align: center;
                        }}

                        h1 {{
                            color: #333;
                        }}

                        #comparison-graph {{
                            margin-top: 20px;
                        }}

                        img {{
                            max-width: 100%;
                            height: auto;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* Optional: Add a subtle box shadow to the image */
                        }}
                    </style>
                </head>
                <body>
                    <h1>Model Comparison Results</h1>
                    <div id="comparison-graph">
                        <img src="{plot_filename}" alt="Model Comparison Plot">
                    </div>
                </body>
                </html>
            ''')

