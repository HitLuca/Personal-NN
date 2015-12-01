package NeuralNetwork;

import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.util.Pair;
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

    List<DoubleMatrix> totals;
    List<DoubleMatrix> activations;

    List<DoubleMatrix> biasDeltas;
    List<DoubleMatrix> weightDeltas;

    public NeuralNetwork(double learningRate, List<Integer> layerSetup, int miniBatchSize) throws FileNotFoundException, UnsupportedEncodingException {
        this.learningRate = learningRate;
        this.layersNumber = layerSetup.size();
        this.miniBatchSize = miniBatchSize;
        this.layerSetup = layerSetup;

        weightDeltas = new ArrayList<>();
        weights = new ArrayList<>();

        for (int l = 0; l < layerSetup.size() - 1; l++) {
            weights.add(new DoubleMatrix().randn(layerSetup.get(l+1), layerSetup.get(l)));
            weightDeltas.add(new DoubleMatrix().zeros(layerSetup.get(l+1), layerSetup.get(l)));
        }

        biases = new ArrayList<>();
        biasDeltas = new ArrayList<>();
        activations = new ArrayList<>();
        totals = new ArrayList<>();

        for (int l = 0; l < layersNumber; l++) {
            activations.add(new DoubleMatrix().zeros(layerSetup.get(l)));
            biases.add(new DoubleMatrix().randn(layerSetup.get(l)));
            biasDeltas.add(new DoubleMatrix().zeros(layerSetup.get(l)));
            totals.add(new DoubleMatrix().zeros(layerSetup.get(l)));
        }
    }

    public void epochTrain(List<Pair<DoubleMatrix, DoubleMatrix>> trainDataset, List<Pair<DoubleMatrix, DoubleMatrix>> validationDataset, List<Pair<DoubleMatrix, DoubleMatrix>> testDataset, int epochs) throws IOException {
        for (int e = 0; e < epochs; e++) {
            System.out.println("Epoch: " + e);
            Collections.shuffle(trainDataset);

            for (int a = 0; a < trainDataset.size(); a+=miniBatchSize) {
                updateMiniBatch(trainDataset.subList(a, a+miniBatchSize));
            }
            if (trainDataset.size()%miniBatchSize!=0) {
                updateMiniBatch(trainDataset.subList(trainDataset.size()-trainDataset.size()%miniBatchSize, trainDataset.size()));
            }

            int success = runTest(testDataset);

            System.out.println("Accuracy: " + String.format( "%.2f",(100.0*success/testDataset.size())) + "%");
        }
    }

    private void updateMiniBatch(List<Pair<DoubleMatrix, DoubleMatrix>> dataset) {
        for (int l = 0; l < layersNumber-1; l++) {
            weightDeltas.set(l, new DoubleMatrix().zeros(layerSetup.get(l+1), layerSetup.get(l)));
            biasDeltas.set(l, new DoubleMatrix().zeros(layerSetup.get(l)));
        }
        biasDeltas.set(layersNumber-1, new DoubleMatrix().zeros(layerSetup.get(layersNumber-1)));

        for (int m=0; m<miniBatchSize; m++) {
            backPropagation(dataset.get(m));
        }

        for (int l=0; l<weights.size(); l++) {
            weights.set(l, weights.get(l).sub(weightDeltas.get(l).mul(learningRate/miniBatchSize)));
        }

        for (int l=0; l<biases.size(); l++) {
            biases.set(l, biases.get(l).sub(biasDeltas.get(l).mul(learningRate/miniBatchSize)));
        }
    }

    private void backPropagation(Pair<DoubleMatrix, DoubleMatrix> input) {
        List<DoubleMatrix> b_deltas = new ArrayList<>();
        List<DoubleMatrix> w_deltas = new ArrayList<>();

        for (int l = 0; l < layersNumber-1; l++) {
            w_deltas.add(new DoubleMatrix().zeros(layerSetup.get(l+1), layerSetup.get(l)));
            b_deltas.add(new DoubleMatrix().zeros(layerSetup.get(l)));
        }
        b_deltas.add(new DoubleMatrix().zeros(layerSetup.get(layersNumber-1)));

        activations.set(0, input.getKey());

        for (int l = 1; l < layersNumber; l++) {
            totals.set(l, weights.get(l-1).mmul(activations.get(l-1)).add(biases.get(l)));
            activations.set(l, sigmoid(totals.get(l)));
        }

        b_deltas.set(layersNumber-1, costDerivative(activations.get(layersNumber-1), input.getValue()).mul(sigmoidPrime(totals.get(layersNumber-1))));
        biasDeltas.set(layersNumber-1, biasDeltas.get(layersNumber-1).add(b_deltas.get(layersNumber-1)));

        for (int l=layersNumber-2; l>0; l--) {
            b_deltas.set(l, (weights.get(l).transpose().mmul(b_deltas.get(l + 1))).mul(sigmoidPrime(totals.get(l))));
            biasDeltas.set(l, biasDeltas.get(l).add(b_deltas.get(l)));
        }

        for (int l=layersNumber-2; l>=0; l--) {
            w_deltas.set(l, b_deltas.get(l+1).mmul(activations.get(l).transpose()));
            weightDeltas.set(l, weightDeltas.get(l).add(w_deltas.get(l)));
        }
    }

    private int runTest(List<Pair<DoubleMatrix, DoubleMatrix>> dataset) {
        int success = 0;
        for (int a = 0; a < dataset.size(); a++) {
            activations.set(0, dataset.get(a).getKey());
            feedForward();
            int index = activations.get(layersNumber-1).argmax();
            if (dataset.get(a).getValue().get(index) == 1.0) {
                success++;
            }
        }
        return success;
    }

    private void feedForward() {
        for (int l = 1; l < layersNumber; l++) {
            totals.set(l, (weights.get(l-1).mmul(activations.get(l-1))).add(biases.get(l)));
            activations.set(l, sigmoid(totals.get(l)));
        }
    }

    private DoubleMatrix costDerivative(DoubleMatrix activations, DoubleMatrix output) {
        return activations.sub(output);
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