package NeuralNetwork;

import NeuralNetwork.Activations.Sigmoid;
import NeuralNetwork.CostFunctions.CostFunction;
import NeuralNetwork.CostFunctions.CrossEntropy;
import NeuralNetwork.CostFunctions.MSE;
import javafx.util.Pair;
import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static Miscellaneous.MiscFunctions.classToMatrix;
import static Miscellaneous.MiscFunctions.createImage;
import static Miscellaneous.NormalizerDenormalizer.normalizeMatrix;

/**
 * Created by hitluca on 07/10/15.
 */
public class NeuralNetwork {
    private double learningRate;
    private double regularization;
    private int miniBatchSize;
    private int layersNumber;
    //private List<Integer> layerSetup;

    private List<NeuronLayer> layers;

    private PrintWriter writer;

    private int epochsWithoutImprovement = 0;
    private double bestEpochCost = Integer.MAX_VALUE;
    private double sumWeights;

    private CostFunction costFunction;

    private boolean dropout;

    public NeuralNetwork(double learningRate, List<Integer> layerSetup, int miniBatchSize, double regularization, int cost, boolean dropout) throws FileNotFoundException, UnsupportedEncodingException {
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.layersNumber = layerSetup.size();
        this.miniBatchSize = miniBatchSize;
        this.dropout = dropout;

        switch (cost) { //0: MSE, 1:CrossEntropy
            case 0: {
                costFunction = new MSE();
                break;
            }
            case 1: {
                costFunction = new CrossEntropy();
                break;
            }
        }

        layers = new ArrayList<>();

        layers.add(new NeuronLayer(layerSetup.get(0), null, false, new Sigmoid()));
        for (int i=1; i<layerSetup.size()-1; i++) {
            layers.add(new NeuronLayer(layerSetup.get(i), layers.get(i - 1), true, new Sigmoid()));
        }
        layers.add(new NeuronLayer(layerSetup.get(layerSetup.size()-1), layers.get(layers.size()-1), true, new Sigmoid()));

        for (int i=0; i<layers.size()-1; i++) {
            layers.get(i).setNextLayer(layers.get(i+1));
        }

        writer = new PrintWriter("output/output.csv", "UTF-8");
    }

    public void epochTrain(List<Pair<DoubleMatrix, DoubleMatrix>> trainDataset, List<Pair<DoubleMatrix, DoubleMatrix>> validationDataset, List<Pair<DoubleMatrix, DoubleMatrix>> testDataset, boolean trainEval, boolean validationEval, boolean testEval) throws IOException {
        Pair<Double, Double> trainResults = null;
        Pair<Double, Double> testResults = null;
        Pair<Double, Double> validationResults = null;

        writer.println("Cost on train data, Cost on test data, Cost on validation data, Accuracy on train data, Accuracy on test data, Accuracy on validation data");

        int e = 0;
        while (true) {
            System.out.println("Epoch: " + e);
            Collections.shuffle(trainDataset);

            if (trainDataset.size() < miniBatchSize) {
                updateMiniBatch(trainDataset, trainDataset.size());
            } else {
                int a;
                for (a = 0; a < trainDataset.size()-miniBatchSize; a += miniBatchSize) {
                    updateMiniBatch(trainDataset.subList(a, a + miniBatchSize), trainDataset.size());
                }
                if (trainDataset.size() % miniBatchSize != 0) {
                    updateMiniBatch(trainDataset.subList(a, trainDataset.size()), trainDataset.size());
                }
            }

            calculatesumWeights();

            if (trainEval) {
                trainResults = evaluate(trainDataset);
                System.out.println("Cost on train data: " + trainResults.getKey());
                writer.print(trainResults.getKey() + ", ");
            }
            if (testEval) {
                testResults = evaluate(testDataset);
                System.out.println("Cost on test data:  " + testResults.getKey());
                writer.print(testResults.getKey() + ", ");
            }
            if (validationEval) {
                validationResults = evaluate(validationDataset);
                System.out.println("Cost on validation data: " + validationResults.getKey());
                writer.print(validationResults.getKey() + ", ");
            }

            if (trainEval) {
                System.out.println("Accuracy on train data: " + String.format("%.2f", trainResults.getValue()) + "%");
                writer.print(String.format("%.2f", trainResults.getValue()) + ", ");
            }
            if (testEval) {
                System.out.println("Accuracy on test data:  " + String.format("%.2f", testResults.getValue()) + "%");
                writer.print(String.format("%.2f", testResults.getValue()) + ", ");
            }
            if (validationEval) {
                System.out.println("Accuracy on validation data: " + String.format("%.2f", validationResults.getValue()) + "%");
                writer.print(String.format("%.2f", validationResults.getValue()) + ", ");
            }

            System.out.println();
            writer.println();
            writer.flush();

            if (validationResults != null) {
                if (earlyStop(validationResults.getKey())) {
                    System.out.println("Early stop!");
                    System.out.println("Best cost: " + bestEpochCost);
                    break;
                }
            }

            //createImages(e);
            e++;

        }
        writer.close();
    }

    private void calculatesumWeights() {
        sumWeights = 0;
        for (int i=1; i<layers.size(); i++) {
            sumWeights += layers.get(i).getSumWeights();
        }
    }

    private void updateMiniBatch(List<Pair<DoubleMatrix, DoubleMatrix>> dataset, int n) {
        for (int m = 0; m < dataset.size(); m++) {
            backPropagation(dataset.get(m));
        }

        for (int i=1; i<layers.size(); i++) {
            layers.get(i).updateWeightsBiases(learningRate, regularization, n, miniBatchSize);
        }
    }

    private void backPropagation(Pair<DoubleMatrix, DoubleMatrix> input) {
        layers.get(0).setActivations(input.getKey());

        for (int l = 1; l < layers.size()-1; l++) {
            layers.get(l).processInput(dropout, false);
        }
        layers.get(layers.size()-1).processInput(false, true);

        layers.get(layers.size()-1).calculateLastLayerBWDeltas(costFunction, input.getValue());
        for (int l = layers.size()-2; l>0; l--) {
            layers.get(l).calculateBWDeltas();
        }
    }

    private Pair<Double, Double> evaluate(List<Pair<DoubleMatrix, DoubleMatrix>> input) {
        double success = 0;
        double cost = 0;

        for (int a = 0; a < input.size(); a++) {
            layers.get(0).setActivations(input.get(a).getKey());
            for (int l = 1; l < layers.size()-1; l++) {
                layers.get(l).processInput(dropout, true);
            }
            layers.get(layers.size()-1).processInput(dropout, true);

            cost += costFunction.singleCost(input.get(a).getValue(), layers.get(layers.size()-1).getActivations());
            if (input.get(a).getValue().get(layers.get(layers.size()-1).getActivations().argmax()) == 1.0) {
                success++;
            }
        }

        cost = costFunction.totalCost(cost, input.size()) + (regularization / (2.0*input.size()) * sumWeights);
        success = 100.0 * success / input.size();

        return new Pair<>(cost, success);
    }

    private boolean earlyStop(double cost) {
        if (epochsWithoutImprovement == 10) {
            return true;
        }

        if (cost < bestEpochCost) {
            bestEpochCost = cost;
            epochsWithoutImprovement = 0;
            learningRate *= 1.1;
        } else {
            epochsWithoutImprovement++;
            learningRate *= 0.5;
        }
        return false;
    }

    private void createImages(int epoch) {
        new File("output/imgs/epoch_" + epoch).mkdirs();
        for (int i = 0; i < 10; i++) {
            layers.get(layersNumber-1).setActivations(classToMatrix(i, 10));
            feedBackward();
            for (int j=0; j<layers.size()-1; j++) {
                new File("output/imgs/epoch_" + epoch + "/layer_" + j).mkdirs();
                createImage(layers.get(j).getActivations(), "output/imgs/epoch_" + epoch + "/layer_" + j + "/class_" + i + ".png");
            }
        }
    }

    private void feedBackward() {
        for (int l=layersNumber-1; l>0; l--) {
            DoubleMatrix x = layers.get(l).calculateBackwardMatrix();
            x = normalizeMatrix(x, x.min(), x.max());
            x.put(x.argmin(), 0.01);
            x.put(x.argmax(), 0.99);
            layers.get(l-1).setActivations(x);
        }
    }
}