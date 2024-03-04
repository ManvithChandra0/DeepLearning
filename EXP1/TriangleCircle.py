import tensorflow as tf


def triangle_area(a, b, c):
    s = (a + b + c) / 2
    area = tf.sqrt(s * (s - a) * (s - b) * (s - c))
    return area


def circle_area(radius):
    r = tf.cast(radius, dtype=tf.float32)
    pi = tf.constant(22 / 7, dtype=tf.float32)
    area = tf.multiply(pi, r, r)
    return area


print("Triangle Area:", triangle_area(5, 12, 13).numpy())

print("Circle Area:", circle_area(6).numpy())
