package NeuralNetwork;

import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.util.Pair;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.io.*;
import java.text.SimpleDateFormat;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Created by hitluca on 07/10/15.
 */
public class NeuralNetwork {
    private double learningRate;
    private double regularization;
    private int miniBatchSize;
    private int layersNumber;
    private List<Integer> layerSetup;

    private List<DoubleMatrix> weights;
    private List<DoubleMatrix> biases;

    private List<DoubleMatrix> totals;
    private List<DoubleMatrix> activations;

    private List<DoubleMatrix> biasDeltas;
    private List<DoubleMatrix> weightDeltas;

    private PrintWriter writer;
    public NeuralNetwork(double learningRate, List<Integer> layerSetup, int miniBatchSize, double regularization) throws FileNotFoundException, UnsupportedEncodingException {
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.layersNumber = layerSetup.size();
        this.miniBatchSize = miniBatchSize;
        this.layerSetup = layerSetup;

        weightDeltas = new ArrayList<>();
        weights = new ArrayList<>();

        for (int l = 0; l < layerSetup.size() - 1; l++) {
            weights.add(new DoubleMatrix().randn(layerSetup.get(l+1), layerSetup.get(l)));
            weightDeltas.add(new DoubleMatrix().zeros(layerSetup.get(l+1), layerSetup.get(l)));
        }

        for (int l = 0; l < weights.size(); l++) {
            double deviation = 1.0 / Math.sqrt(weights.get(l).columns);
            weights.get(l).mul(deviation);
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

        writer = new PrintWriter("output/output.csv", "UTF-8");
    }

    public void epochTrain(List<Pair<DoubleMatrix, DoubleMatrix>> trainDataset, List<Pair<DoubleMatrix, DoubleMatrix>> validationDataset, List<Pair<DoubleMatrix, DoubleMatrix>> testDataset, int epochs) throws IOException {
        Pair<Double, Double> trainResults;
        Pair<Double, Double> testResults;
        double pastCost = 0;

        writer.println("Cost on train data, Cost on test data, Accuracy on train data, Accuracy on test data, Learning rate");

        for (int e = 0; e < epochs; e++) {
            System.out.println("Epoch: " + e);
            Collections.shuffle(trainDataset);

            for (int a = 0; a < trainDataset.size(); a+=miniBatchSize) {
                updateMiniBatch(trainDataset.subList(a, a+miniBatchSize), trainDataset.size());
            }
            if (trainDataset.size()%miniBatchSize!=0) {
                updateMiniBatch(trainDataset.subList(trainDataset.size()-trainDataset.size()%miniBatchSize, trainDataset.size()), trainDataset.size());
            }

            trainResults = evaluate(trainDataset);
            testResults = evaluate(testDataset);

            System.out.println("Cost on train data: " + trainResults.getKey());
            System.out.println("Cost on test data:  " + testResults.getKey());

            System.out.println("Accuracy on train data: " + String.format( "%.2f", trainResults.getValue()) + "%");
            System.out.println("Accuracy on test data:  " + String.format( "%.2f", testResults.getValue()) + "%");

            System.out.println("Learning rate: " + String.format( "%.4f", learningRate));

            writer.println(trainResults.getKey() + ", " + testResults.getKey() + ", " + String.format( "%.2f", trainResults.getValue()) + ", " + String.format( "%.2f", testResults.getValue()) + ", " + String.format( "%.4f", learningRate));
            writer.flush();

            if (pastCost!=0) {
                if (pastCost<trainResults.getKey()) {
                    learningRate *= 0.95;
                }
                else
                {
                    learningRate += 0.05;
                }
            }
            pastCost = trainResults.getKey();

            //saveData();
        }
        writer.close();
    }

    private void saveData() throws IOException {
        ObjectMapper mapper = new ObjectMapper();

        SimpleDateFormat formatter = new SimpleDateFormat("dd-MM-yyyy_hh-mm-ss");
        Date date = new Date();
        mapper.writeValue(new File("output/" + formatter.format(date) + "_Network.l33t"), this.getClass());
    }

    private void updateMiniBatch(List<Pair<DoubleMatrix, DoubleMatrix>> dataset, int n) {
        for (int l = 0; l < layersNumber-1; l++) {
            weightDeltas.set(l, new DoubleMatrix().zeros(layerSetup.get(l+1), layerSetup.get(l)));
            biasDeltas.set(l, new DoubleMatrix().zeros(layerSetup.get(l)));
        }
        biasDeltas.set(layersNumber-1, new DoubleMatrix().zeros(layerSetup.get(layersNumber-1)));

        for (int m=0; m<miniBatchSize; m++) {
            backPropagation(dataset.get(m));
        }

        for (int l=0; l<weights.size(); l++) {
            weights.set(l, (weights.get(l).mul(1-(learningRate*regularization/n))).sub(weightDeltas.get(l).mul(learningRate/miniBatchSize)));
        }

        for (int l=0; l<biases.size(); l++) {
            biases.set(l, biases.get(l).sub(biasDeltas.get(l).mul(learningRate/miniBatchSize)));
        }
    }

    private void backPropagation(Pair<DoubleMatrix, DoubleMatrix> input) {
        List<DoubleMatrix> b_deltas = new ArrayList<>();

        for (int l = 0; l < layersNumber-1; l++) {
            b_deltas.add(new DoubleMatrix().zeros(layerSetup.get(l)));
        }
        b_deltas.add(new DoubleMatrix().zeros(layerSetup.get(layersNumber-1)));

        activations.set(0, input.getKey());

        for (int l = 1; l < layersNumber; l++) {
            totals.set(l, weights.get(l-1).mmul(activations.get(l-1)).add(biases.get(l)));
            activations.set(l, sigmoid(totals.get(l)));
        }

        b_deltas.set(layersNumber-1, costDerivative(activations.get(layersNumber-1), input.getValue()));
        biasDeltas.set(layersNumber-1, biasDeltas.get(layersNumber-1).add(b_deltas.get(layersNumber-1)));

        for (int l=layersNumber-2; l>0; l--) {
            b_deltas.set(l, (weights.get(l).transpose().mmul(b_deltas.get(l + 1))).mul(sigmoidPrime(totals.get(l))));
            biasDeltas.set(l, biasDeltas.get(l).add(b_deltas.get(l)));
        }

        for (int l=layersNumber-2; l>=0; l--) {
            weightDeltas.set(l, weightDeltas.get(l).add(b_deltas.get(l+1).mmul(activations.get(l).transpose())));
        }
    }

    private double calculateCost(Pair<DoubleMatrix, DoubleMatrix> input) {
        double total = 0;
        for (int i=0; i<activations.get(layersNumber-1).rows; i++) {
            total += input.getValue().get(i)*Math.log(activations.get(layersNumber-1).get(i)) + (1 - input.getValue().get(i))*Math.log(1-activations.get(layersNumber-1).get(i));
        }
        return total;
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

    private Pair<Double, Double> evaluate(List<Pair<DoubleMatrix, DoubleMatrix>> input) {
        double success = 0;
        double cost = 0;

        for (int a = 0; a < input.size(); a++) {
            activations.set(0, input.get(a).getKey());
            feedForward();
            cost += calculateCost(input.get(a));
            int index = activations.get(layersNumber-1).argmax();
            if (input.get(a).getValue().get(index) == 1.0) {
                success++;
            }
        }

        cost = -1.0/input.size()*cost;
        success = 100.0*success/input.size();

        return new Pair<>(cost, success);
    }
}