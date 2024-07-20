import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Example of a basic computation
a = tf.constant(2)
b = tf.constant(3)
c = a + b

print("Result of addition:", c.numpy())