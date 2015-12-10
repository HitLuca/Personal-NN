import DatasetLoaders.IrisLoader;
import DatasetLoaders.MnistLoader;
import NeuralNetwork.*;
import javafx.util.Pair;
import org.jblas.DoubleMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by hitluca on 07/10/15.
 */
public class Main {
    private static Scanner scanner;

    public static void main(String[] args) throws IOException, InterruptedException {
        scanner = new Scanner(System.in);

        System.out.println("Enter parameters:");
        System.out.print("MiniBatchSize: ");
        int miniBatchSize = (int) getUserInput();
        System.out.print("LearningRate: ");
        double learningRate = getUserInput();
        System.out.print("Regularization: ");
        double regularization = getUserInput();

        List<Integer> layerSetup = new ArrayList<>();

        List<Pair<DoubleMatrix, DoubleMatrix>> trainDataset = null;
        List<Pair<DoubleMatrix, DoubleMatrix>> validationDataset = null;
        List<Pair<DoubleMatrix, DoubleMatrix>> testDataset = null;

        System.out.println("Choose dataset");
        System.out.println("(1) mnist");
        System.out.println("(2) iris");
        int choice = (int) getUserInput();
        switch (choice) {
            case 1: {
                layerSetup.add(784);
                System.out.print("Hidden neurons: ");
                int i= (int) getUserInput();
                if (i!=0) {
                    layerSetup.add(i);
                }
                layerSetup.add(10);

                MnistLoader MnistLoader = new MnistLoader("data/mnist/mnist_train.csv");
                System.out.println("Importing train and validation data...");
                MnistLoader.importData(true);
                trainDataset = MnistLoader.getTrain();
                validationDataset = MnistLoader.getValidation();

                MnistLoader = new MnistLoader("data/mnist/mnist_test.csv");
                System.out.println("Importing test data...");
                MnistLoader.importData(false);
                testDataset = MnistLoader.getTest();
                break;
            }
            case 2: {
                layerSetup.add(4);
                System.out.print("Hidden neurons: ");
                int i= (int) getUserInput();
                if (i!=0) {
                    layerSetup.add(i);
                }
                layerSetup.add(3);

                IrisLoader irisLoader = new IrisLoader("data/iris/iris.data");
                System.out.println("Importing data...");
                irisLoader.importData(true);
                trainDataset = irisLoader.getTrain();
                validationDataset = irisLoader.getValidation();
                testDataset = irisLoader.getTest();
                break;
            }
            default: {
                return;
            }
        }

        NeuralNetwork neuralNetwork = new NeuralNetwork(learningRate, layerSetup, miniBatchSize, regularization, 1, true);
        neuralNetwork.epochTrain(trainDataset, validationDataset, testDataset, true, true, true);
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
            }
        }
        return d;
    }
}
