import DataImportExport.CSVLoader;
import NeuralNetwork.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by hitluca on 07/10/15.
 */
public class Main {
    static Scanner scanner;

    static int epochs = 0;
    static int miniBatchSize = 0;
    static double learningRate;
    static double momentum;
    static NeuralNetwork neuralNetwork = null;

    static List<DatasetElement> trainDataset = new ArrayList<>();
    static List<DatasetElement> testDataset = new ArrayList<>();

    public static void main(String[] args) throws IOException, InterruptedException {
        scanner = new Scanner(System.in);

        System.out.println("Enter parameters:");
        System.out.print("Epochs: ");
        epochs = Integer.parseInt(scanner.nextLine());
        System.out.print("MiniBatchSize: ");
        miniBatchSize = Integer.parseInt(scanner.nextLine());
        System.out.print("LearningRate: ");
        learningRate = Double.parseDouble(scanner.nextLine());
//        System.out.print("Momentum: ");
//        momentum = Double.parseDouble(scanner.nextLine());

        List<Integer> layerSetup = new ArrayList<>();
        layerSetup.add(784);
        System.out.print("Hidden neurons: ");
        layerSetup.add(Integer.parseInt(scanner.nextLine()));
        layerSetup.add(10);

        CSVLoader csvLoader = new CSVLoader("data/mnist_train.csv");
        System.out.println("Importing data...");
        csvLoader.importData();
        trainDataset = csvLoader.getDataset();

        csvLoader = new CSVLoader("data/mnist_test.csv");
        System.out.println("Importing data...");
        csvLoader.importData();
        testDataset = csvLoader.getDataset();

        neuralNetwork = new NeuralNetwork(learningRate, layerSetup, miniBatchSize);
        neuralNetwork.epochTrain(trainDataset, epochs, testDataset);
    }
}
