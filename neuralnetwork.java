/**
 * Created by Reece Martin on 2017-05-08.
 */

import java.util.Scanner;

public class neuralnetwork{

    // Variables for correct (trained) input and output values
    private double[] input1c;
    private double[] input2c;
    private double[] outputc;

    // Variables for neuron values

    // Inputs are limited to the range of zero to one inclusive
    private double input1;
    private double input2;

    private double hiddenLayerA1Neuron;
    private double hiddenLayerA2Neuron;
    private double hiddenLayerB1Neuron;
    private double hiddenLayerB2Neuron;
    private double bias1 = 1;
    private double bias2 = 1;
    private double bias3 = 1;

    // Weights (layer type-layer-neuron number-weight)
    //Weights for Hidden Layer A Neuron 1
    private double ha1a;
    private double ha1b;
    private double ha1bias;

    //Weights for Hidden Layer A Neuron 2
    private double ha2a;
    private double ha2b;
    private double ha2bias;

    //Weights for Hidden Layer B Neuron 1
    private double hb1a;
    private double hb1b;
    private double hb1bias;

    //Weights for Hidden Layer B Neuron 2
    private double hb2a;
    private double hb2b;
    private double hb2bias;

    //Weights for Output Neuron
    private double o1a;
    private double o1b;
    private double o1bias;

    public neuralnetwork(){

        // Ideal Values (To Be Used in Training)
        input1c = new double[4];
        input2c = new double[4];
        outputc = new double[4];

        // Initialize Weights to Random Values??

    }


    public void trainNetwork(){

        Scanner read = new Scanner(System.in);

        // Prompts user to enter the maximum acceptable error rate which will be used to train the network.
        System.out.println("Enter ideal max error rate:");
        double threshold = read.nextDouble();

        double error = 0;

        // (MSE) - Mean Squared Error Calculation

        // Initial test allows error to be initialized properly.
        // for-loop runs through each training condition and we return the average of the error rates.
        for(int i = 0; i < 4; i++)
        {
            error += Math.pow(outputc[i] - testNetwork(input1c[i], input2c[i]), 2); // Make sure to verify error calculation
        }
        error /= 4;

        //Calculate Initial and set error

        // Loop runs until we meet the user set threshold.
        while(error > threshold);

        // for loop runs through each training condition.

    }


    public double testNetwork(double inputf1, double inputf2){

        //Calculation for first layer of hidden neurons
        double hiddenLayerA1 = activationFunction(inputf1 * ha1a * inputf2 * ha1b * bias1 * ha1bias);
        double hiddenLayerA2 = activationFunction(inputf1 * ha2a * inputf2 * ha2b * bias1 * ha2bias);

        //Calculation for second layer of hidden neurons
        double hiddenLayerB1 = activationFunction(hiddenLayerA1 * hb1a * hiddenLayerA2 * hb1b * bias2 * hb1bias);
        double hiddenLayerB2 = activationFunction(hiddenLayerA1 * hb2a * hiddenLayerA2 * hb2b * bias2 * hb2bias);

        //Calculation for output
        return activationFunction(hiddenLayerB1);

    }

    /*
    public static double activationFunction(double value){

    }
    */

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
    }
}

