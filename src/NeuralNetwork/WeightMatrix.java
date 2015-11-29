package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by hitluca on 06/11/15.
 */
public class WeightMatrix {
    private List<List<Double>> weights;
    private int rows;
    private int columns;

    public WeightMatrix(int rows, int columns, boolean setZero) {
        Random random = new Random();
        this.rows = rows;
        this.columns = columns;

        weights = new ArrayList<>();

        for (int i = 0; i < rows; i++) {
            weights.add(new ArrayList<>());
            for (int j = 0; j < columns; j++) {
                double d;
                if (setZero) {
                    d = 0;
                } else {
                    d = random.nextGaussian();
                }
                weights.get(i).add(d);
            }
        }
    }

    public WeightMatrix(List<List<Double>> weights) {
        this.rows = weights.size();
        this.columns = weights.get(0).size();

        this.weights = new ArrayList<>(weights);
    }

    public List<List<Double>> getWeights() {
        return weights;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public double getWeight(int i, int j) {
        return weights.get(i).get(j);
    }

    public void setWeight (int i, int j, double d) {
        weights.get(i).set(j, d);
    }
}
