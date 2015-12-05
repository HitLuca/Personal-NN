package NeuralNetwork.CostFunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import static NeuralNetwork.Activations.Sigmoid.sigmoidPrime;

/**
 * Created by hitluca on 05/12/15.
 */
public class MSE implements CostFunction{
    public double singleCost(DoubleMatrix expectedOutput, DoubleMatrix activations) {
        return MatrixFunctions.pow(expectedOutput.sub(activations), 2).sum();
    }

    public double totalCost(double cost, int n) {
        return 1.0 / (2*n) * cost;
    }

    public DoubleMatrix costDerivative(DoubleMatrix activations, DoubleMatrix output, DoubleMatrix totals) {
        return activations.sub(output).mul(sigmoidPrime(totals));
    }
}
