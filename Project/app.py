from flask import Flask, render_template
import preprocessing as pp
from Model import Model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# Creating an instance (Flask)
app = Flask(__name__)
obj = pp.PreprocessData()
dir_path = "/Users/manvithchandra/Desktop/DeepLearning/EXP/lfw"
leaf_df, train, labels = obj.preprocess(dir_path)
tr_gen, tt_gen, va_gen = obj.generate_train_test_images(leaf_df, train, labels)
input_shape = (128, 128, 3)
ms = Model()

# Render/route - Normal route
@app.route("/")
def sample():
    return render_template("index.html")

@app.route('/model/simple_ANN')
def model_function1():
    # Simple Model
    model_simple = ms.simple_model(input_shape=input_shape, optimizer="adam")

    # Fit the model
    history = model_simple.fit(tr_gen, validation_data=va_gen, epochs=10)

    # Plotting
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training and Validation Accuracy: Simple ANN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('static/images/compare.jpg')  # Save the plot
    plt.close()  # Close the plot to free memory
    return render_template('index.html', model_comparison_graph_url='/static/images/compare.jpg')



if __name__ == "__main__":
    app.run(debug=True)
