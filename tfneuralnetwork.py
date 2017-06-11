# Import Tensorflow Module
import tensorflow as tf

# Import Numpy Module
import numpy as np

# Setup Data in three lists, all variables are cast as Numpy-float32 which is functional with Tensorflow
input_a = [np.float32(0.0), np.float32(0.0), np.float32(1.0), np.float32(1.0)]
input_b = [np.float32(0.0), np.float32(1.0), np.float32(0.0), np.float32(1.0)],
outputs = [np.float32(0.0), np.float32(1.0), np.float32(1.0), np.float32(0.0)]

# Initialization of our weights and biases
weight_a = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
weight_b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

err_total = 0

# Loop for checking error across all possible configurations
for i in range(3):
    output = weight_a * input_a[i]+ biases
    err_total += tf.square(output - outputs[i])

# Total Error is the mean of the error on all four possible configurations
err_total /= 4

# Training
optimization = tf.train.GradientDescentOptimizer(learning_rate=0.5)
training = optimization.minimize(err_total)

# Initialization
init = tf.initialize_all_variables()
runtime = tf.Session()
runtime.run(init)

# Loop to control Training Iterations 
for i in range(100000):
    runtime.run(training)
    if i % 1000 == 0:
        print(i, runtime.run(weight_a), runtime.run(weight_b), runtime.run(biases))

# Plan to print the result of the trained network on each configuration here
print(weight_a * 1 + weight_b * 1 + biases)
print(weight_a * 0 + weight_b * 0 + biases)
print(weight_a * 1 + weight_b * 0 + biases)
print(weight_a * 0 + weight_b * 1 + biases)
