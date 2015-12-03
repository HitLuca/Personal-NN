import DataImportExport.CSVLoader;
import NeuralNetwork.*;
import javafx.util.Pair;
import org.jblas.DoubleMatrix;

import javax.xml.crypto.Data;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by hitluca on 07/10/15.
 */
public class Main {
    private static Scanner scanner;

    private static int epochs = 0;
    private static int miniBatchSize = 0;
    private static double learningRate;
    private static double regularization;
    private static double momentum;
    private static NeuralNetwork neuralNetwork = null;

    private static List<Pair<DoubleMatrix, DoubleMatrix>> trainDataset;
    private static List<Pair<DoubleMatrix, DoubleMatrix>> testDataset;
    private static List<Pair<DoubleMatrix, DoubleMatrix>> validationDataset;

    public static void main(String[] args) throws IOException, InterruptedException {
        scanner = new Scanner(System.in);

        System.out.println("Enter parameters:");
        System.out.print("MiniBatchSize: ");
        miniBatchSize = (int) getUserInput();
        System.out.print("LearningRate: ");
        learningRate = getUserInput();
        System.out.print("Regularization: ");
        regularization = getUserInput();
//        System.out.print("Momentum: ");
//        momentum = Double.parseDouble(scanner.nextLine());

        List<Integer> layerSetup = new ArrayList<>();
        layerSetup.add(784);
        System.out.print("Hidden neurons: ");
        layerSetup.add((int) getUserInput());
        layerSetup.add(10);

        CSVLoader csvLoader = new CSVLoader("data/mnist_train.csv");
        System.out.println("Importing train and validation data...");
        csvLoader.importData(true);
        trainDataset = csvLoader.getDataset();
        validationDataset = csvLoader.getValidation();

        csvLoader = new CSVLoader("data/mnist_test.csv");
        System.out.println("Importing test data...");
        csvLoader.importData(false);
        testDataset = csvLoader.getDataset();

        neuralNetwork = new NeuralNetwork(learningRate, layerSetup, miniBatchSize, regularization);
        neuralNetwork.epochTrain(trainDataset, validationDataset, testDataset);
    }

    private static double getUserInput() {
        boolean success = false;
        String s;
        Double d = null;
        while(!success) {
            s = scanner.nextLine();
            try {
                d = Double.parseDouble(s);
                success = true;
            } catch(Exception e){
                System.out.println("L'input non è valido");
            };
        }
        return d;
    }
}
