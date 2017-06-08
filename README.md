# binaryneuralnet
 A elementary feedforward multilayer perceptron Neural Network designed to pattern match basic binary operators such as And Or and Xor. Successfully converges when "hinted" values are used.

 Inspired by lectures from Jeff Heaton's videos on Neural Networks found here: https://www.youtube.com/channel/UCR1-GEpyOPzT2AO4D_eifdw
 Based on Wikipedias pages on Perceptrons, Gradient Descent, and Backpropagation.

# What is does:
The Binary Operators AND, OR, and XOR are the basis for propositional logic which is fundamental to a great deal of topics in mathematics and computer science. When many university students take their first course in propositional logic and discrete mathematics they learn the truth tables for these operators, this Neural Network does just that. When given a truth table the Neural Network cannot correctly apply the operators in it's initial (untrained) state however, following proper training the Neural Network can "learn" the correct patterns for the different operators and apply them with a high degree of accuracy.  

# How it works:
The basis of any Neural Network is a series of Neurons and Weights (connections with synaptic strengths between neurons)[see diagram]. A Neural Network may have i input neurons and o output neurons and some number of hidden neurons arranged in layers between the input and output neurons. When a Neural Network is initialized it's weights are set to random floating-point values which when given an input a is unlikely to produce a desired input b however, through the process of training ... the neural networks weights can be adjusted such that the network produces an output which is very near to the desired output b.
