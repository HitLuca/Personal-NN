package NeuralNetwork;

import javafx.util.Pair;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static Miscellaneous.MiscFunctions.classToMatrix;
import static Miscellaneous.MiscFunctions.createImage;
import static Miscellaneous.NormalizerDenormalizer.denormalize;
import static Miscellaneous.NormalizerDenormalizer.normalizeMatrix;
import static NeuralNetwork.Activations.Logit.logit;
import static NeuralNetwork.Activations.Sigmoid.sigmoid;
import static NeuralNetwork.Activations.Sigmoid.sigmoidPrime;
import static NeuralNetwork.CostFunctions.CrossEntropy.calculateCost;
import static NeuralNetwork.CostFunctions.CrossEntropy.costDerivative;

/**
 * Created by hitluca on 07/10/15.
 */
public class NeuralNetwork {
    private double learningRate;
    private double initialLearningRate;
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

    private int epochsWithoutImprovement = 0;
    private double bestEpochAccuracy = 0;
    private double sumWeights;

    public NeuralNetwork(double learningRate, List<Integer> layerSetup, int miniBatchSize, double regularization) throws FileNotFoundException, UnsupportedEncodingException {
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        this.regularization = regularization;
        this.layersNumber = layerSetup.size();
        this.miniBatchSize = miniBatchSize;
        this.layerSetup = layerSetup;

        weightDeltas = new ArrayList<>();
        weights = new ArrayList<>();

        for (int l = 0; l < layerSetup.size() - 1; l++) {
            weights.add(new DoubleMatrix().randn(layerSetup.get(l + 1), layerSetup.get(l)));
            weightDeltas.add(new DoubleMatrix().zeros(layerSetup.get(l + 1), layerSetup.get(l)));
        }

        for (int l = 0; l < weights.size(); l++) {
            double deviation = 1.0 / Math.sqrt(weights.get(l).columns);
            weights.set(l, weights.get(l).mul(deviation));
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

    public void epochTrain(List<Pair<DoubleMatrix, DoubleMatrix>> trainDataset, List<Pair<DoubleMatrix, DoubleMatrix>> validationDataset, List<Pair<DoubleMatrix, DoubleMatrix>> testDataset) throws IOException {
        Pair<Double, Double> trainResults;
        Pair<Double, Double> testResults;
        Pair<Double, Double> validationResults;

        writer.println("Cost on train data, Cost on test data, Cost on validation data, Accuracy on train data, Accuracy on test data, Accuracy on validation data");

        int e = 0;
        while (true) {
            System.out.println("Epoch: " + e);
            Collections.shuffle(trainDataset);

            if (trainDataset.size() < miniBatchSize) {
                updateMiniBatch(trainDataset, trainDataset.size());
            } else {
                for (int a = 0; a < trainDataset.size(); a += miniBatchSize) {
                    updateMiniBatch(trainDataset.subList(a, a + miniBatchSize), trainDataset.size());
                }
                if (trainDataset.size() % miniBatchSize != 0) {
                    updateMiniBatch(trainDataset.subList(trainDataset.size() - (trainDataset.size() % miniBatchSize), trainDataset.size()), trainDataset.size());
                }
            }

            calculatesumWeights();

            trainResults = evaluate(trainDataset);
            testResults = evaluate(testDataset);
            validationResults = evaluate(validationDataset);

            System.out.println("Cost on train data: " + trainResults.getKey());
            System.out.println("Cost on test data:  " + testResults.getKey());
            System.out.println("Cost on validation data:  " + validationResults.getKey());

            System.out.println("Accuracy on train data: " + String.format("%.2f", trainResults.getValue()) + "%");
            System.out.println("Accuracy on test data:  " + String.format("%.2f", testResults.getValue()) + "%");
            System.out.println("Accuracy on validation data: " + String.format("%.2f", validationResults.getValue()) + "%");

            System.out.println();
            writer.println(trainResults.getKey() + ", " + testResults.getKey() + ", " + validationResults.getKey() + ", " + String.format("%.2f", trainResults.getValue()) + ", " + String.format("%.2f", testResults.getValue()) + ", " + String.format("%.2f", validationResults.getValue()));
            writer.flush();

            if (earlyStop(validationResults.getValue())) {
                System.out.println("Early stop!");
                System.out.println("Best accuracy: " + bestEpochAccuracy);
                break;
            }

            //createImages(e);
            e++;

        }
        writer.close();
    }

    private void calculatesumWeights() {
        sumWeights = 0;
        for (int l = 0; l < weights.size(); l++) {
            for (int i = 0; i < weights.get(l).rows; i++) {
                for (int j = 0; j < weights.get(l).columns; j++) {
                    sumWeights += Math.pow(weights.get(l).get(i, j), 2);
                }
            }
        }
    }

    private void updateMiniBatch(List<Pair<DoubleMatrix, DoubleMatrix>> dataset, int n) {
        for (int l = 0; l < layersNumber - 1; l++) {
            weightDeltas.set(l, new DoubleMatrix().zeros(layerSetup.get(l + 1), layerSetup.get(l)));
            biasDeltas.set(l, new DoubleMatrix().zeros(layerSetup.get(l)));
        }
        biasDeltas.set(layersNumber - 1, new DoubleMatrix().zeros(layerSetup.get(layersNumber - 1)));

        for (int m = 0; m < miniBatchSize; m++) {
            backPropagation(dataset.get(m));
        }

        for (int l = 0; l < weights.size(); l++) {
            weights.set(l, (weights.get(l).mul(1.0 - (learningRate * regularization / n))).sub(weightDeltas.get(l).mul(learningRate / miniBatchSize)));
        }

        for (int l = 0; l < biases.size(); l++) {
            biases.set(l, biases.get(l).sub(biasDeltas.get(l).mul(learningRate / miniBatchSize)));
        }
    }

    private void backPropagation(Pair<DoubleMatrix, DoubleMatrix> input) {
        List<DoubleMatrix> b_deltas = new ArrayList<>();

        for (int l = 0; l < layersNumber; l++) {
            b_deltas.add(new DoubleMatrix().zeros(layerSetup.get(l)));
        }

        activations.set(0, input.getKey());

        for (int l = 1; l < layersNumber; l++) {
            totals.set(l, (weights.get(l - 1).mmul(activations.get(l - 1))).add(biases.get(l)));
            activations.set(l, sigmoid(totals.get(l)));
        }

        b_deltas.set(layersNumber - 1, costDerivative(activations.get(layersNumber - 1), input.getValue()));
        biasDeltas.set(layersNumber - 1, biasDeltas.get(layersNumber - 1).add(b_deltas.get(layersNumber - 1)));

        for (int l = layersNumber - 2; l > 0; l--) {
            b_deltas.set(l, ((weights.get(l).transpose()).mmul(b_deltas.get(l + 1))).mul(sigmoidPrime(totals.get(l))));
            biasDeltas.set(l, biasDeltas.get(l).add(b_deltas.get(l)));
        }

        for (int l = layersNumber - 2; l >= 0; l--) {
            weightDeltas.set(l, weightDeltas.get(l).add(b_deltas.get(l + 1).mmul(activations.get(l).transpose())));
        }
    }

    private void feedForward() {
        for (int l = 1; l < layersNumber; l++) {
            totals.set(l, (weights.get(l - 1).mmul(activations.get(l - 1))).add(biases.get(l)));
            activations.set(l, sigmoid(totals.get(l)));
        }
    }

    private Pair<Double, Double> evaluate(List<Pair<DoubleMatrix, DoubleMatrix>> input) {
        double success = 0;
        double cost = 0;

        for (int a = 0; a < input.size(); a++) {
            activations.set(0, input.get(a).getKey());
            feedForward();
            cost += calculateCost(input.get(a).getValue(), activations.get(layersNumber-1));
            if (input.get(a).getValue().get(activations.get(layersNumber - 1).argmax()) == 1.0) {
                success++;
            }
        }

        cost = (-1.0 / input.size())*cost + regularization / (2.0*input.size()) * sumWeights;
        success = 100.0 * success / input.size();

        return new Pair<>(cost, success);
    }

    private boolean earlyStop(double success) {
        if (epochsWithoutImprovement == 20) {
            return true;
        }

        if (success > bestEpochAccuracy) {
            bestEpochAccuracy = success;
            epochsWithoutImprovement = 0;
        } else {
            epochsWithoutImprovement++;
            if (epochsWithoutImprovement == 5) {
                learningRate /= 10;
                miniBatchSize = 1;
                epochsWithoutImprovement = 0;
                if (initialLearningRate / learningRate > 1000) {
                    return true;
                }
            }
        }
        System.out.println();
        return false;
    }

    private void createImages(int epoch) {
        new File("output/imgs/" + epoch).mkdirs();
        for (int i = 0; i < 10; i++) {
            activations.set(layersNumber - 1, classToMatrix(i, 10));
            feedBackward();
            createImage(activations.get(0), "output/imgs/" + epoch + "/" + i + ".png");
        }
    }

    private void feedBackward() {
        for (int l=layersNumber-1; l>0; l--) {
            totals.set(l, logit(activations.get(l)).sub(biases.get(l)));
            DoubleMatrix x = Solve.solveLeastSquares(weights.get(l-1), totals.get(l));
            x = normalizeMatrix(x, x.min(), x.max());
            x.put(x.argmin(), 0.01);
            x.put(x.argmax(), 0.99);
            activations.set(l-1, x);
        }
    }
}