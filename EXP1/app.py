from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf

app = Flask(__name__)

def calareaoftriangle(a, b, c):
    s = (a + b + c) / 2.0
    area = tf.sqrt(s * (s - a) * (s - b) * (s - c))
    return area.numpy()

def calareaofcircle(r):
    r = tf.cast(r, dtype=tf.float32)
    pi = tf.constant(22 / 7, dtype=tf.float32)
    area = tf.multiply(pi, r, r)
    return area.numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_triangle', methods=['POST'])
def calculate_triangle():
    side_a = float(request.form['sideA'])
    side_b = float(request.form['sideB'])
    side_c = float(request.form['sideC'])
    result = calareaoftriangle(side_a, side_b, side_c)
    return redirect(url_for('result', result=result))

@app.route('/calculate_circle', methods=['POST'])
def calculate_circle():
    radius = float(request.form['radius'])
    result = calareaofcircle(radius)
    return redirect(url_for('result', result=result))

@app.route('/result')
def result():
    result = request.args.get('result', type=float)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=8001)
