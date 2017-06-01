/**
 * Created by Reece Martin on 2017-05-08.
 */

import java.util.Scanner;

public class neuralnetwork{

    // Variables for correct (trained) input and output values
    public double[] input1c;
    public double[] input2c;
    public double[] outputc;

    // Variables for neuron values

    // Inputs are limited to the range of zero to one inclusive
    public double input1;
    public double input2;

    public double hiddenLayerA1Neuron;
    public double hiddenLayerA2Neuron;
    public double bias1 = 1;
    public double bias2 = 1;
    public double bias3 = 1;

    // Weights (layer type-layer-neuron number-weight)
    // Weights for Hidden Layer A Neuron 1
    public double ha1a;
    public double ha1b;
    public double ha1bias;

    // Weights for Hidden Layer A Neuron 2
    public double ha2a;
    public double ha2b;
    public double ha2bias;

    // Weights for Output Neuron
    public double o1a;
    public double o1b;
    public double o1bias;

    public neuralnetwork(){

        // Ideal Values (To Be Used in Training)
        input1c = new double[4];
        input2c = new double[4];
        outputc = new double[4];
    }

    public void weightGenerator(){

        // Initializes all weights to random initial values, since these values are not constrained we multiply to
        // increase the variability.
        ha1a = Math.random();
        ha1b = Math.random();
        ha1bias = Math.random();
        ha2a = Math.random();
        ha2b = Math.random();
        ha2bias = Math.random();
        o1a = Math.random();
        o1b = Math.random();
        o1bias = Math.random();
    }

    public void trainNetwork(){

        Scanner read = new Scanner(System.in);

        // Prompts user to enter the maximum acceptable error rate which will be used to train the network.
        System.out.println("Enter ideal max error rate:");
        double threshold = read.nextDouble();

        // Prompts user to enter the learning rate which will be used to train the network.
        System.out.println("Enter learning rate (Tip: try 0.5):");
        double learningRate = read.nextDouble();
        // Learning Rate is sometimes referred to as Epsilon.

        // Prompts user to enter the momentum which will be used to train the network.
        System.out.println("Enter momentum (Tip: try 0.2):");
        double momentum = read.nextDouble();
        // Momentum is sometimes referred to as Alpha.

        double error = 0;

        // (MSE) - Mean Squared Error Calculation

        // Initial test allows error to be initialized properly.
        // for-loop runs through each training condition and we return the average of the error rates.
        for(int i = 0; i < 4; i++) {
            error += Math.pow(outputc[i] - testNetwork(input1c[i], input2c[i]), 2);
        }
        error /= 4;

        // Loop runs until we meet the user set threshold.
        int x = 0;
        while(x < 10000) {

            // The Previous Weight Deltas
            double alpha[] = {0, 0, 0, 0, 0, 0, 0, 0, 0} ;

            // for loop runs through each training condition.
            for (int i = 0; i < 4; i++) {

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

                // Backpropagation

                double ha1aDelta = -learningRate * ha1aGradient + momentum * alpha[0];
                        ha1a += ha1aDelta;
                alpha[0] = ha1aDelta;

                double ha1bDelta = -learningRate * ha1bGradient + momentum * alpha[1];
                        ha1b += ha1bDelta;
                alpha[1] = ha1bDelta;

                double ha1biasDelta = -learningRate * ha1biasGradient + momentum * alpha[2];
                        ha1bias += ha1biasDelta;
                alpha[2] = ha1biasDelta;

                double ha2aDelta = -learningRate * ha2aGradient + momentum * alpha[3];
                        ha2a += ha2aDelta;
                alpha[3] = ha2aDelta;

                double ha2bDelta = -learningRate * ha2bGradient + momentum * alpha[4];
                        ha2b += ha2bDelta;
                alpha[4] = ha2bDelta;

                double ha2biasDelta = -learningRate * ha2biasGradient + momentum * alpha[5];
                        ha2bias += ha2biasDelta;
                alpha[5] = ha2biasDelta;

                double o1aDelta =  -learningRate * o1aGradient + momentum * alpha[6];
                        o1a += o1aDelta;
                alpha[6] = o1aDelta;

                double o1bDelta =  -learningRate * o1bGradient + momentum * alpha[7];
                        o1b += o1bDelta;
                alpha[7] = o1bDelta;

                double o1biasDelta = -learningRate * o1biasGradient + momentum * alpha[8];
                        o1bias += o1biasDelta;
                alpha[0] = ha1aDelta;


            }

            for(int i = 0; i < 4; i++) {
                error += Math.pow(outputc[i] - testNetwork(input1c[i], input2c[i]), 2);
            }
            error /= 4;
            System.out.println(error);
            x++;
        }
    }


    public double testNetwork(double inputf1, double inputf2){

        // Calculation for first layer of hidden neurons
        double hiddenLayerA1 = activationFunction(inputf1 * ha1a + inputf2 * ha1b + bias1 * ha1bias);
        double hiddenLayerA2 = activationFunction(inputf1 * ha2a + inputf2 * ha2b + bias1 * ha2bias);

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
    // truth table and so only binary operator emulation can be achieved with this neural network implementation

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

        net.weightGenerator();

        System.out.println("1, 0");
        System.out.println(net.testNetwork(1.0, 0.0));

        System.out.println("1, 1");
        System.out.println(net.testNetwork(1.0, 1.0));

        System.out.println("0, 0");
        System.out.println(net.testNetwork(0.0, 0.0));

        System.out.println("0, 1");
        System.out.println(net.testNetwork(0.0, 1.0));

        net.trainNetwork();

        System.out.println("1, 0");
        System.out.println(net.testNetwork(1.0, 0.0));

        System.out.println("1, 1");
        System.out.println(net.testNetwork(1.0, 1.0));

        System.out.println("0, 0");
        System.out.println(net.testNetwork(0.0, 0.0));

        System.out.println("0, 1");
        System.out.println(net.testNetwork(0.0, 1.0));

    }
}

