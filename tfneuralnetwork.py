# Import Tensorflow
import tensorflow as tf
import numpy as np

# Setup Data in a two dimensional list - the first two lists are inputs the third list is outputs
input_a = [np.float32(0.0), np.float32(0.0), np.float32(1.0), np.float32(1.0)]
input_b = [np.float32(0.0), np.float32(1.0), np.float32(0.0), np.float32(1.0)]
outputs = [np.float32(0.0), np.float32(1.0), np.float32(1.0), np.float32(0.0)]


weight_a = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
weight_b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

err_total = 0
for i in range(3):

    output = weight_a * input_a[i] + weight_b * input_b[i] + biases
    err_total += tf.square(output - outputs[i])

err_total /= 4

optimization = tf.train.GradientDescentOptimizer(learning_rate=0.5)
training = optimization.minimize(err_total)

init = tf.global_variables_initializer()
runtime = tf.Session()
runtime.run(init)

for i in range(10000):
    # This number of iterations seems sufficient
    runtime.run(training)
    if i % 1000 == 0:
        print(i, runtime.run(weight_a), runtime.run(weight_b), runtime.run(biases))
        
# Need to turn tensors into type ndarray then to type list then to type float32
print(np.float32((runtime.run(weight_a)).tolist()[0]) * 1 +
      np.float32((runtime.run(weight_b)).tolist()[0]) * 1 +
       (np.float32((runtime.run(biases)).tolist()[0])))
print(np.float32((runtime.run(weight_a)).tolist()[0]) * 0 +
      np.float32((runtime.run(weight_b)).tolist()[0]) * 0 +
       (np.float32((runtime.run(biases)).tolist()[0])))
print(np.float32((runtime.run(weight_a)).tolist()[0]) * 1 +
      np.float32((runtime.run(weight_b)).tolist()[0]) * 0 +
       (np.float32((runtime.run(biases)).tolist()[0])))
print(np.float32((runtime.run(weight_a)).tolist()[0]) * 0 +
      np.float32((runtime.run(weight_b)).tolist()[0]) * 1 +
       (np.float32((runtime.run(biases)).tolist()[0])))
