package NeuralNetwork.CostFunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import static NeuralNetwork.Activations.Sigmoid.sigmoidPrime;

/**
 * Created by hitluca on 05/12/15.
 */
public interface CostFunction {
    double singleCost(DoubleMatrix expectedOutput, DoubleMatrix activations);

    double totalCost(double cost, int n);

    DoubleMatrix costDerivative(DoubleMatrix activations, DoubleMatrix output, DoubleMatrix totals);
}
