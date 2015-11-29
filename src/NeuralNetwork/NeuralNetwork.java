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

    List<DoubleMatrix> weights;
    List<DoubleMatrix> biases;

    List<DoubleMatrix> totals;
    List<DoubleMatrix> activations;
    List<DoubleMatrix> errors;

    public NeuralNetwork(double learningRate, List<Integer> layerSetup, int miniBatchSize) throws FileNotFoundException, UnsupportedEncodingException {
        this.learningRate = learningRate;
        this.layersNumber = layerSetup.size();
        this.miniBatchSize = miniBatchSize;

        weights = new ArrayList<>();
        for (int l = 0; l < layerSetup.size() - 1; l++) {
            weights.add(new DoubleMatrix().randn(layerSetup.get(l+1), layerSetup.get(l)));
        }

        biases = new ArrayList<>();
        activations = new ArrayList<>();
        errors = new ArrayList<>();
        totals = new ArrayList<>();

        for (int l = 0; l < layerSetup.size(); l++) {
            biases.add(new DoubleMatrix().randn(layerSetup.get(l)));
        }

        for (int l = 0; l < layerSetup.size(); l++) {
            activations.add(new DoubleMatrix().zeros(layerSetup.get(l)));
            errors.add(new DoubleMatrix().zeros(layerSetup.get(l)));
            totals.add(new DoubleMatrix().zeros(layerSetup.get(l)));
        }

    }

    public void epochTrain(List<DatasetElement> trainDataset, int epochs, List<DatasetElement> testDataset) throws IOException {
        double costTrain;
        double costTest;
        double trainError;
        double testError;

        for (int e = 0; e < epochs; e++) {
            System.out.println("Epoch: " + e);
            trainError = 0;

            Collections.shuffle(trainDataset);

            for (int a = 0; a < trainDataset.size(); a++) {
                //Setting inputs
                for (int i = 0; i < trainDataset.get(a).getInputs().size(); i++) {
                    activations.set(0, new DoubleMatrix(trainDataset.get(a).getInputs()));
                }

                //Forward pass
                for (int l = 1; l < layersNumber; l++) {
                    totals.set(l, (weights.get(l-1).mmul(activations.get(l-1))).add(biases.get(l)));
                    activations.set(l, sigmoid(totals.get(l)));
                }

                //Output errors
                errors.set(layersNumber-1, (activations.get(layersNumber-1).sub(new DoubleMatrix(trainDataset.get(a).getOutputs()))).mul(sigmoidPrime(totals.get(layersNumber-1))));
                for (int i=0; i<errors.get(layersNumber-1).rows; i++) {
                    trainError += Math.pow(activations.get(layersNumber-1).get(i) - trainDataset.get(a).getOutputs().get(i), 2);
                }

                //Backpropagating the error
                for (int l = layersNumber - 2; l > 0; l--) {
                    errors.set(l, (weights.get(l).transpose().mmul(errors.get(l+1))).mul(sigmoidPrime(totals.get(l))));

                }

                //Updating the biases
                for (int l = 1; l < layersNumber - 1; l++) {
                    biases.set(l, biases.get(l).sub(errors.get(l).mul(learningRate)));
                }

                //Updating the weights
                for (int l = 1; l < activations.size(); l++) {
                    for (int i=0; i<weights.get(l-1).rows; i++) {
                        for (int j=0; j<weights.get(l-1).columns; j++) {
                            weights.get(l-1).put(i, j, weights.get(l-1).get(i, j) - learningRate * activations.get(l-1).get(j) * errors.get(l).get(i));
                        }
                    }
                }
            }

            int success = 0;
            testError = 0;

            for (int a = 0; a < testDataset.size(); a++) {
                //Setting inputs
                for (int i = 0; i < trainDataset.get(a).getInputs().size(); i++) {
                    activations.set(0, new DoubleMatrix(testDataset.get(a).getInputs()));
                }

                //Forward pass
                for (int l = 1; l < layersNumber; l++) {
                    totals.set(l, (weights.get(l-1).mmul(activations.get(l-1))).add(biases.get(l)));
                    activations.set(l, sigmoid(totals.get(l)));
                }

                //Output errors
                for (int i=0; i<errors.get(layersNumber-1).rows; i++) {
                    testError += Math.pow(activations.get(layersNumber-1).get(i) - testDataset.get(a).getOutputs().get(i), 2);
                }

                int index = activations.get(layersNumber-1).argmax();

                if (testDataset.get(a).getOutputs().get(index) == 1.0) {
                    success++;
                }
            }
            costTrain = (1.0 / (2 * trainDataset.size())) * trainError;
            costTest = (1.0 / (2 * testDataset.size())) * testError;

            System.out.println("Cost on training: " + costTrain);
            System.out.println("Cost on test: " + costTest);
            System.out.println("Accuracy: " + String.format( "%.2f",(100.0*success/testDataset.size())) + "%");
            //writeData();
        }
    }

    private void writeData() throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File("output/output.json"), weights);
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
        return sigmoid(input).mul(ones.subi(sigmoid(input)));
    }
}