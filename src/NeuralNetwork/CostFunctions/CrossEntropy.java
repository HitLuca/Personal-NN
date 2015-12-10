package NeuralNetwork.CostFunctions;

import NeuralNetwork.Activations.ActivationFunction;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by hitluca on 05/12/15.
 */
public class CrossEntropy implements CostFunction {
    public double singleCost(DoubleMatrix expectedOutput, DoubleMatrix activations) {
        DoubleMatrix onesInput = new DoubleMatrix().ones(expectedOutput.rows);
        DoubleMatrix onesActivations = new DoubleMatrix().ones(activations.rows);

        for (int i=0; i<activations.rows; i++) {
            if (activations.get(i)==0) {
                activations.put(i, 0.0000000001);
            } else if (activations.get(i)==1) {
                activations.put(i, 0.9999999999);
            }
        }

        DoubleMatrix a = expectedOutput.mul(MatrixFunctions.log(activations));
        DoubleMatrix b = (onesInput.sub(expectedOutput)).mul(MatrixFunctions.log(onesActivations.sub(activations)));
        return a.add(b).sum();
    }

    public double totalCost(double cost, int n) {
        return (-1.0 / n) * cost;
    }

    public DoubleMatrix costDerivative(DoubleMatrix activations, DoubleMatrix output, DoubleMatrix totals, ActivationFunction activation) {
        return activations.sub(output);
    }
}
