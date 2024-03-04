# app.py

from flask import Flask, render_template
from models import DeepANN
from preprocessing import preprocess_data
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)


def load_and_preprocess_data(dir_path):
    data_preprocessor = preprocess_data()

    # Preprocess the data
    image_df, train, labels = data_preprocessor.preprocess(dir_path)

    # Generate train, test, and validation image generators
    tr_gen, tt_gen, va_gen = data_preprocessor.generate_train_test_images(image_df, train, labels)

    return tr_gen, tt_gen, va_gen

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    # Adjust the directory path based on your project structure
    dir_path = r"train"

    # Load and preprocess data
    tr_gen, tt_gen, va_gen = load_and_preprocess_data(dir_path)

    # Create an instance of the DeepANN class
    deep_ann = DeepANN()

    # Create multiple instances of your model with different optimizers (you can customize this)
    models = [
        deep_ann.simple_model(input_shape=(29, 29, 3), optimizer="adam"),
        deep_ann.simple_model(input_shape=(29, 29, 3), optimizer="sgd"),
        deep_ann.simple_model(input_shape=(29, 29, 3), optimizer="rmsprop")
    ]

    # Train and compare models, get the plot filename
    plot_filename = deep_ann.compare_models(models, tr_gen, va_gen, epochs=5)

    # Save the comparison graph as an HTML file
    comparison_html_path = 'templates/comparison.html'
    deep_ann.save_comparison_graph(comparison_html_path, plot_filename)

    return render_template('comparison.html', comparison_html_path=comparison_html_path)

if __name__ == '__main__':
    app.run(debug=True)
