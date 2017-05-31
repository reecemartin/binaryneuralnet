/**
 * Created by Reece Martin on 2017-05-08.
 */

import java.util.Scanner;

public class neuralnetwork{

    // Variables for correct (trained) input and output values.
    private double[] input1c;
    private double[] input2c;
    private double[] outputc;

    // Variables for neuron values.
    // Inputs are limited to the range of zero to one inclusive.
    private double input1;
    private double input2;

    private double hiddenLayerA1Neuron;
    private double hiddenLayerA2Neuron;
    private double bias1 = 1;
    private double bias2 = 1;
    private double bias3 = 1;

    // Weights (layer type-layer-neuron number-weight)
    // Weights for Hidden Layer A Neuron 1
    private double ha1a;
    private double ha1b;
    private double ha1bias;

    // Weights for Hidden Layer A Neuron 2
    private double ha2a;
    private double ha2b;
    private double ha2bias;

    // Weights for Output Neuron
    private double o1a;
    private double o1b;
    private double o1bias;

    public neuralnetwork(){

        // Ideal Values (To Be Used in Training)
        input1c = new double[4];
        input2c = new double[4];
        outputc = new double[4];
    }

    public void weightGenerator(){

        // Initializes all weights to random initial values, since these values are not constrained we multiply to
        // increase the variability.
        ha1a = Math.random() * 10000 - 5000;
        ha1b = Math.random() * 10000 - 5000;
        ha1bias = Math.random() * 10000 - 5000;
        ha2a = Math.random() * 10000 - 5000;
        ha2b = Math.random() * 10000 - 5000;
        ha2bias = Math.random() * 10000 - 5000;
        o1a = Math.random() * 10000 - 5000;
        o1b = Math.random() * 10000 - 5000;
        o1bias = Math.random() * 10000 - 5000;
    }

    public void trainNetwork(){

        Scanner read = new Scanner(System.in);
        // Prompts user to enter the maximum acceptable error rate which will be used to train the network.
        System.out.println("Enter ideal max error rate:");
        double threshold = read.nextDouble();

        // Prompts user to enter the learning rate which will be used to train the network.
        System.out.println("Enter learning rate:");
        double learningRate = read.nextDouble();

        double error = 0;

        // (MSE) - Mean Squared Error Calculation

        // Initial test allows error to be initialized properly.
        // for-loop runs through each training condition and we return the average of the error rates.
        for(int i = 0; i < 4; i++){
            error += Math.pow(outputc[i] - testNetwork(input1c[i], input2c[i]), 2);
            // Make sure to verify error calculation
        }
        error /= 4;

        // Calculate Initial and set error
        // NODE DELTAS FRM OUTPUT INWARDS
        // Calculate Gradients

        // Loop runs until we meet the user set threshold.
        while(error > threshold);

        // for loop runs through each training condition.
        for(int i = 0; i < 4; i++) {
            // Calculation for first layer of hidden neurons
            double hiddenLayerA1 = activationFunction(input1c[i] * ha1a + input2c[i] * ha1b + bias1 * ha1bias);
            double hiddenLayerA2 = activationFunction(input1c[i] * ha2a + input2c[i] * ha2b + bias1 * ha2bias);

            // Calculation for output
            double outputResult = activationFunction(hiddenLayerA1 * o1a + hiddenLayerA2 * o1b + bias3 * o1bias);

            // Gradient Calculations

            // Output Neuron
            double outputError = outputResult - outputc[i];
            double outputLayerDelta = -outputError * activationFunctionDerivative(
                    hiddenLayerA1 * o1a + hiddenLayerA2 * o1b + bias3 * o1bias);

            // Hidden Neuron 1
            double hidden1LayerDelta = activationFunctionDerivative(
                    input1c[i] * ha1a + input2c[i] * ha1b + bias1 * ha1bias) * o1a * outputLayerDelta;

            // Hidden Neuron 2
            double hidden2LayerDelta = activationFunctionDerivative(
                    input1c[i] * ha2a + input2c[i] * ha2b + bias1 * ha2bias) * o1b * outputLayerDelta;

            // Weight Gradients for Hidden Layer A Neuron 1
            double ha1aGradient = input1c[i] * hidden1LayerDelta;
            double ha1bGradient = input2c[i] * hidden1LayerDelta;
            double ha1biasGradient = hidden1LayerDelta;

            // Weight Gradients for Hidden Layer A Neuron 2
            double ha2aGradient = input1c[i] * hidden2LayerDelta;
            double ha2bGradient = input2c[i] * hidden2LayerDelta;
            double ha2biasGradient = hidden2LayerDelta;

            // Weight Gradients for Output Neuron
            double o1aGradient = activationFunction(
                    input1c[i] * ha1a + input2c[i] * ha1b + bias1 * ha1bias) * outputLayerDelta;
            double o1bGradient = activationFunction(
                    input1c[i] * ha2a + input2c[i] * ha2b + bias1 * ha2bias) * outputLayerDelta;
            double o1biasGradient = outputLayerDelta;

        }
    }


    public double testNetwork(double inputf1, double inputf2){
        // Calculation for first layer of hidden neurons
        double hiddenLayerA1 = activationFunction(input1c[i] * ha1a + input2c[i] * ha1b + bias1 * ha1bias);
        double hiddenLayerA2 = activationFunction(input1c[i] * ha2a + input2c[i] * ha2b + bias1 * ha2bias);

        // Calculation for output
        double outputResult = activationFunction(hiddenLayerA1 * o1a + hiddenLayerA2 * o1b + bias3 * o1bias);
        return outputResult;
    }


    public static double activationFunction(double value){
        // Sigmoid Activation Function
        return 1 / (1 + Math.pow(Math.E, -value));
    }

    public static  double activationFunctionDerivative(double value){
        // Derivative of Sigmoid Function
        return (Math.pow(Math.E, -value)) / Math.pow((1 + Math.pow(Math.E, -value)), 2);
    }



    // Method used for user to input data to be trained to the neural network - data is entered in the form of a 3 X 4
    // truth table and so only binary operator emulation can be achieved with this neural network implementation.

    public void inputTrainingData(){
        Scanner read = new Scanner(System.in);
        for (int i = 0; i < 4; i++) {
            System.out.println("Enter the first input for row # " + (i + 1));
            input1c[i] = read.nextInt();
            System.out.println("Enter the second input for row # " + (i + 1));
            input2c[i] = read.nextInt();
            System.out.println("Enter the output for row #" + (i + 1));
            outputc[i] = read.nextInt();
        }
    }

    public static void main(String[] args) {
        // Initializes a new neural network
        neuralnetwork net = new neuralnetwork();

        // Calls method to prompt user to input data to be trained to the neural network
        net.inputTrainingData();
    }
}

