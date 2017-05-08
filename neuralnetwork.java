/**
 * Created by Reece Martin on 2017-05-08.
 */

import java.util.Scanner;

public class neuralnetwork{

    //Variables for correct (trained) input and output values
    private double[] input1c;
    private double[] input2c;
    private double[] outputc;


    public neuralnetwork(){
        input1c = new double[4];
        input2c = new double[4];
        outputc = new double[4];
    }

    /*
    public void trainNetwork{

    }

    public void testNetwork{

    }
    */

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
        neuralnetwork net = new neuralnetwork();
        net.inputTrainingData();
    }
}

