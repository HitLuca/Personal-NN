package NeuralNetwork.CostFunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by hitluca on 05/12/15.
 */
public class CrossEntropy implements CostFunction {
    public double singleCost(DoubleMatrix expectedOutput, DoubleMatrix activations) {
        DoubleMatrix onesInput = new DoubleMatrix().ones(expectedOutput.rows);
        DoubleMatrix onesActivations = new DoubleMatrix().ones(activations.rows);

        DoubleMatrix a = expectedOutput.mul(MatrixFunctions.log(activations));
        DoubleMatrix b = (onesInput.sub(expectedOutput)).mul(MatrixFunctions.log(onesActivations.sub(activations)));
        return a.add(b).sum();
    }

    public double totalCost(double cost, int n) {
        return -1.0 / n * cost;
    }

    public DoubleMatrix costDerivative(DoubleMatrix activations, DoubleMatrix output, DoubleMatrix totals) {
        return activations.sub(output);
    }
}
