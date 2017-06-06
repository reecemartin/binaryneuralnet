/**
 * Created by Reece Martin on 2017-05-08.
 */

import java.util.Scanner;

public class neuralnetwork {

    // Variables for correct (trained) input and output values
    private double[] input1c, input2c, outputc;

    // Variables for neuron values

    // Inputs are limited to the range of zero to one inclusive
    private double input1, input2;

    private double bias1 = 1;
    private double bias2 = 1;
    private double bias3 = 1;

    // Weights (layer type-layer-neuron number-weight)
    // Weights for Hidden Layer A Neuron 1
    private double ha1a, ha1b, ha1bias;

    // Weights for Hidden Layer A Neuron 2
    private double ha2a, ha2b, ha2bias;

    // Weights for Output Neuron
    private double o1a, o1b, o1bias;

    // Constructor
    public neuralnetwork() {

        // Ideal Values (To Be Used in Training)
        input1c = new double[4];
        input2c = new double[4];
        outputc = new double[4];
    }


    // Method used to generate random weights
    public void weightGenerator() {

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


    // Method used to train Network
    public void trainNetwork() {

        Scanner read = new Scanner(System.in);
        System.out.println("\nExpress or Custom Settings e/c?");
        String settingsSelector = read.nextLine();

        double threshold, learningRate, momentum;

        if (settingsSelector.equals("e")){

            threshold = 0.0001;
            learningRate = 0.5;
            momentum = 0.2;

        }else {

            // Prompts user to enter the maximum acceptable error rate which will be used to train the network.
            System.out.println("\nEnter ideal max error rate:");
            threshold = read.nextDouble();

            // Prompts user to enter the learning rate which will be used to train the network.
            System.out.println("\nEnter learning rate:");
            learningRate = read.nextDouble();
            // Learning Rate is sometimes referred to as Epsilon.

            // Prompts user to enter the momentum which will be used to train the network.
            System.out.println("\nEnter momentum:");
            momentum = read.nextDouble();
            // Momentum is sometimes referred to as Alpha.
        }

        double error = 0;

        // (MSE) - Mean Squared Error Calculation
        // Initial test allows error to be initialized properly.
        // for-loop runs through each training condition and we return the average of the error rates.
        for (int i = 0; i < 4; i++) {
            error += Math.pow(outputc[i] - testNetwork(input1c[i], input2c[i]), 2);
        }

        error /= 4;
        int counter = 0;

        // Loop runs until we meet the user set threshold.
        while (error > threshold) {

            // The Previous Weight Deltas
            double alpha[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

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
                double ha1aDelta = learningRate * ha1aGradient + momentum * alpha[0];
                ha1a += ha1aDelta;
                alpha[0] = ha1aDelta;

                double ha1bDelta = learningRate * ha1bGradient + momentum * alpha[1];
                ha1b += ha1bDelta;
                alpha[1] = ha1bDelta;

                double ha1biasDelta = learningRate * ha1biasGradient + momentum * alpha[2];
                ha1bias += ha1biasDelta;
                alpha[2] = ha1biasDelta;

                double ha2aDelta = learningRate * ha2aGradient + momentum * alpha[3];
                ha2a += ha2aDelta;
                alpha[3] = ha2aDelta;

                double ha2bDelta = learningRate * ha2bGradient + momentum * alpha[4];
                ha2b += ha2bDelta;
                alpha[4] = ha2bDelta;

                double ha2biasDelta = learningRate * ha2biasGradient + momentum * alpha[5];
                ha2bias += ha2biasDelta;
                alpha[5] = ha2biasDelta;

                double o1aDelta = learningRate * o1aGradient + momentum * alpha[6];
                o1a += o1aDelta;
                alpha[6] = o1aDelta;

                double o1bDelta = learningRate * o1bGradient + momentum * alpha[7];
                o1b += o1bDelta;
                alpha[7] = o1bDelta;

                double o1biasDelta = learningRate * o1biasGradient + momentum * alpha[8];
                o1bias += o1biasDelta;
                alpha[0] = ha1aDelta;


            }

            for (int i = 0; i < 4; i++) {
                error += Math.pow(outputc[i] - testNetwork(this.input1c[i], this.input2c[i]), 2);
            }
            error /= 4;
            System.out.println("Error:" + error);
            counter++;

        }

        System.out.println("\nReached " + threshold + " Mean Squared Error, in " + counter + " iterations.\n");
    }

    // Method for Network Evaluation
    public double testNetwork(double inputf1, double inputf2) {

        // Calculation for first layer of hidden neurons
        double hiddenLayerA1 = activationFunction(inputf1 * this.ha1a + inputf2 * this.ha1b + this.bias1 * this.ha1bias);
        double hiddenLayerA2 = activationFunction(inputf1 * this.ha2a + inputf2 * this.ha2b + this.bias1 * this.ha2bias);

        // Calculation for output
        double outputResult = activationFunction(hiddenLayerA1 * this.o1a + hiddenLayerA2 * this.o1b + this.bias3 * this.o1bias);
        return outputResult;

    }


    // Method used for user to input data to be trained to the neural network - data is entered in the form of a 3 X 4
    // truth table and so only binary operator emulation can be achieved with this neural network implementation
    public void inputTrainingData() {
        Scanner read = new Scanner(System.in);
        boolean escape = false;

        while (!escape) {
            System.out.println("Enter XOR for Exclusive Or, OR for Or, AND for And, or C for custom:");
            String selection = read.nextLine();

            if (selection.equals("OR")) {
                this.input1c[0] = 1;
                this.input1c[1] = 1;
                this.input1c[2] = 0;
                this.input1c[3] = 0;
                this.input2c[0] = 1;
                this.input2c[1] = 0;
                this.input2c[2] = 1;
                this.input2c[3] = 0;
                this.outputc[0] = 1;
                this.outputc[1] = 1;
                this.outputc[2] = 1;
                this.outputc[3] = 0;
                System.out.println("Or Operator Selected");
                escape = true;
            }

            if (selection.equals("AND")) {
                this.input1c[0] = 1;
                this.input1c[1] = 1;
                this.input1c[2] = 0;
                this.input1c[3] = 0;
                this.input2c[0] = 1;
                this.input2c[1] = 0;
                this.input2c[2] = 1;
                this.input2c[3] = 0;
                this.outputc[0] = 1;
                this.outputc[1] = 0;
                this.outputc[2] = 0;
                this.outputc[3] = 0;
                System.out.println("And Operator Selected");
                escape = true;
            }

            if (selection.equals("XOR")) {
                this.input1c[0] = 1;
                this.input1c[1] = 1;
                this.input1c[2] = 0;
                this.input1c[3] = 0;
                this.input2c[0] = 1;
                this.input2c[1] = 0;
                this.input2c[2] = 1;
                this.input2c[3] = 0;
                this.outputc[0] = 0;
                this.outputc[1] = 1;
                this.outputc[2] = 1;
                this.outputc[3] = 0;
                System.out.println("Exclusive Or Operator Selected");
                escape = true;
            }

            if (selection.equals("C")) {

                System.out.println("Custom Input Selected");
                for (int i = 0; i < 4; i++) {
                    System.out.println("Enter the first input for row # " + (i + 1));
                    this.input1c[i] = read.nextInt();
                    System.out.println("Enter the second input for row # " + (i + 1));
                    this.input2c[i] = read.nextInt();
                    System.out.println("Enter the output for row #" + (i + 1));
                    this.outputc[i] = read.nextInt();
                }
                escape = true;
            }
        }
    }


    // Activation Functions
    public static double activationFunction(double value) {
        // Sigmoid Activation Function
        return 1 / (1 + Math.pow(Math.E, -value));
    }

    public static double activationFunctionDerivative(double value) {
        // Derivative of Sigmoid Function
        return (Math.pow(Math.E, -value)) / Math.pow((1 + Math.pow(Math.E, -value)), 2);
    }

    public void printWeights(){
        System.out.println("Weights:");

        System.out.println("\nHidden Layer Neuron 1:");
        System.out.println("Weight A: " + this.ha1a);
        System.out.println("Weight B: " + this.ha1b);
        System.out.println("Weight Bias: " + this.ha1bias);
        System.out.println("\nHidden Layer Neuron 2:");
        System.out.println("Weight A: " + this.ha2a);
        System.out.println("Weight B: " + this.ha2b);
        System.out.println("Weight Bias: " + this.ha2bias);
        System.out.println("\nOutput Neuron:");
        System.out.println("Weight A: " + this.o1a);
        System.out.println("Weight B: " + this.o1b);
        System.out.println("Weight Bias: " + this.o1bias);
    }


    public void printResults(){
        System.out.println("1, 0");
        System.out.println(this.testNetwork(1.0, 0.0));

        System.out.println("1, 1");
        System.out.println(this.testNetwork(1.0, 1.0));

        System.out.println("0, 0");
        System.out.println(this.testNetwork(0.0, 0.0));

        System.out.println("0, 1");
        System.out.println(this.testNetwork(0.0, 1.0));
        System.out.println("\n");
    }


    public static void main(String[] args) {
        String runSim = "y";

        while(runSim.equals("y")) {

            if (runSim.equals("n")){
                break;
            }
            Scanner read = new Scanner(System.in);
            System.out.println("\nRun Simulation y/n?");
            runSim = read.nextLine();
            if (runSim.equals("n")){
                break;
            }

            // Initializes a new neural network
            neuralnetwork net = new neuralnetwork();

            // Calls method to prompt user to input data to be trained to the neural network
            net.inputTrainingData();

            net.weightGenerator();

            // Prints out Pre=Training Weights
            System.out.println("\nGet weights y/n?");
            String getWeightResponse = read.nextLine();

            if (getWeightResponse.equals("y")) {
                net.printWeights();
            }

            System.out.println("\nRaw Output:\n");
            net.printResults();

            net.trainNetwork();

            System.out.println("\nTrained Output:\n");
            net.printResults();

            // Prints out Post-Training Weights
            System.out.println("\nGet weights y/n?");
            getWeightResponse = read.nextLine();

            if (getWeightResponse.equals("y")) {
                net.printWeights();
            }
        }
    }
}
