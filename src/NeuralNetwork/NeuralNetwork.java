package NeuralNetwork;

import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by hitluca on 07/10/15.
 */
public class NeuralNetwork {
    private double learningRate;
    private int miniBatchSize;
    private int layersNumber;
    private List<Integer> layerSetup;

    List<DoubleMatrix> weights;
    List<DoubleMatrix> biases;

    List<List<DoubleMatrix>> totals;
    List<List<DoubleMatrix>> activations;
    List<List<DoubleMatrix>> errors;

    List<DoubleMatrix> miniBatchErrors;
    List<DoubleMatrix> miniBatchActivations;

    public NeuralNetwork(double learningRate, List<Integer> layerSetup, int miniBatchSize) throws FileNotFoundException, UnsupportedEncodingException {
        this.learningRate = learningRate;
        this.layersNumber = layerSetup.size();
        this.miniBatchSize = miniBatchSize;
        this.layerSetup = layerSetup;

        weights = new ArrayList<>();
        for (int l = 0; l < layerSetup.size() - 1; l++) {
            weights.add(new DoubleMatrix().randn(layerSetup.get(l+1), layerSetup.get(l)));
        }

        biases = new ArrayList<>();
        activations = new ArrayList<>();
        errors = new ArrayList<>();
        totals = new ArrayList<>();

        miniBatchErrors = new ArrayList<>();
        miniBatchActivations = new ArrayList<>();

        for (int l = 0; l < layerSetup.size(); l++) {
            biases.add(new DoubleMatrix().randn(layerSetup.get(l)));
            miniBatchErrors.add(new DoubleMatrix().zeros(layerSetup.get(l)));
            miniBatchActivations.add(new DoubleMatrix().zeros(layerSetup.get(l)));
        }

        for (int m=0; m<miniBatchSize; m++) {
            activations.add(new ArrayList<>());
            errors.add(new ArrayList<>());
            totals.add(new ArrayList<>());
            for (int l = 0; l < layerSetup.size(); l++) {
                activations.get(m).add(new DoubleMatrix().zeros(layerSetup.get(l)));
                errors.get(m).add(new DoubleMatrix().zeros(layerSetup.get(l)));
                totals.get(m).add(new DoubleMatrix().zeros(layerSetup.get(l)));
            }
        }


    }

    public void epochTrain(List<DatasetElement> trainDataset, List<DatasetElement> validationDataset, List<DatasetElement> testDataset, int epochs) throws IOException {
        double costTrain;
        double costTest;
        double trainError;
        double testError;

        for (int e = 0; e < epochs; e++) {
            System.out.println("Epoch: " + e);
            trainError = 0;

            Collections.shuffle(trainDataset);

            for (int a = 0; a < trainDataset.size(); a+=miniBatchSize) {
                if (a%5000==0) {
                    System.out.println(a);
                }

                for (int l = 0; l < layerSetup.size(); l++) {
                    miniBatchErrors.set(l, new DoubleMatrix().zeros(layerSetup.get(l)));
                    miniBatchActivations.set(l, new DoubleMatrix().zeros(layerSetup.get(l)));
                }

                for (int m=0; m<miniBatchSize; m++) {
                    //Setting inputs
                    for (int i = 0; i < trainDataset.get(a+m).getInputs().size(); i++) {
                        activations.get(m).set(0, new DoubleMatrix(trainDataset.get(a+m).getInputs()));
                        miniBatchActivations.set(0, miniBatchActivations.get(0).add(activations.get(m).get(0)));
                    }

                    //Forward pass
                    for (int l = 1; l < layersNumber; l++) {
                        totals.get(m).set(l, (weights.get(l-1).mmul(activations.get(m).get(l-1))).add(biases.get(l)));
                        activations.get(m).set(l, sigmoid(totals.get(m).get(l)));
                        miniBatchActivations.set(l, miniBatchActivations.get(l).add(activations.get(m).get(l)));
                    }

                    //Output errors
                    errors.get(m).set(layersNumber-1, (activations.get(m).get(layersNumber-1).sub(new DoubleMatrix(trainDataset.get(a+m).getOutputs()))).mul(sigmoidPrime(totals.get(m).get(layersNumber-1))));
                    miniBatchErrors.set(layersNumber-1, miniBatchErrors.get(layersNumber-1).add(errors.get(m).get(layersNumber-1)));
                    for (int i=0; i<errors.get(m).get(layersNumber-1).rows; i++) {
                        trainError += Math.pow(activations.get(m).get(layersNumber-1).get(i) - trainDataset.get(a+m).getOutputs().get(i), 2);
                    }

                    //Backpropagating the error
                    for (int l = layersNumber - 2; l > 0; l--) {
                        errors.get(m).set(l, (weights.get(l).transpose().mmul(errors.get(m).get(l+1))).mul(sigmoidPrime(totals.get(m).get(l))));
                        miniBatchErrors.set(l, miniBatchErrors.get(l).add(errors.get(m).get(l)));
                    }
                }

                //Updating the biases
                for (int l = 1; l < layersNumber; l++) {
                    biases.set(l, biases.get(l).sub(miniBatchErrors.get(l).mul(learningRate/miniBatchSize)));
                }

                //Updating the weights
                for (int l = 0; l < weights.size(); l++) {
                    weights.set(l, weights.get(l).sub((miniBatchErrors.get(l+1).mmul(miniBatchActivations.get(l).transpose()).mul(learningRate/miniBatchSize))));
                }
            }

            int success = 0;
            testError = 0;

            for (int a = 0; a < testDataset.size(); a++) {
                //Setting inputs
                for (int i = 0; i < testDataset.get(a).getInputs().size(); i++) {
                    activations.get(0).set(0, new DoubleMatrix(testDataset.get(a).getInputs()));
                }

                //Forward pass
                for (int l = 1; l < layersNumber; l++) {
                    totals.get(0).set(l, (weights.get(l-1).mmul(activations.get(0).get(l-1))).add(biases.get(l)));
                    activations.get(0).set(l, sigmoid(totals.get(0).get(l)));
                }

                //Output errors
                for (int i=0; i<errors.get(0).get(layersNumber-1).rows; i++) {
                    testError += Math.pow(activations.get(0).get(layersNumber-1).get(i) - testDataset.get(a).getOutputs().get(i), 2);
                }

                //Check result
                int index = activations.get(0).get(layersNumber-1).argmax();
                if (testDataset.get(a).getOutputs().get(index) == 1.0) {
                    success++;
                }
            }

            //Calculating costs
            costTrain = (1.0 / (2 * trainDataset.size())) * trainError;
            costTest = (1.0 / (2 * testDataset.size())) * testError;

            System.out.println("Cost on training: " + costTrain);
            System.out.println("Cost on test: " + costTest);
            System.out.println("Accuracy: " + String.format( "%.2f",(100.0*success/testDataset.size())) + "%");
        }
    }

    private DoubleMatrix sigmoid(DoubleMatrix input) {
        DoubleMatrix result = new DoubleMatrix().copy(input);
        DoubleMatrix ones = DoubleMatrix.ones(result.rows, result.columns);
        result.muli(-1);
        MatrixFunctions.expi(result);
        result.addi(1);
        return ones.divi(result);
    }

    private DoubleMatrix sigmoidPrime(DoubleMatrix input) {
        DoubleMatrix ones = DoubleMatrix.ones(input.rows, input.columns);
        return sigmoid(input).mul(ones.sub(sigmoid(input)));
    }
}