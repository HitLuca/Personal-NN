package NeuralNetwork.CostFunctions;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by hitluca on 05/12/15.
 */
public class CrossEntropy {
    public static double calculateCost(DoubleMatrix expectedOutput, DoubleMatrix activations) {
        DoubleMatrix onesInput = new DoubleMatrix().ones(expectedOutput.rows);
        DoubleMatrix onesActivations = new DoubleMatrix().ones(activations.rows);

        DoubleMatrix a = expectedOutput.mul(MatrixFunctions.log(activations));
        DoubleMatrix b = (onesInput.sub(expectedOutput)).mul(MatrixFunctions.log(onesActivations.sub(activations)));
        return a.add(b).sum();
    }

    public static DoubleMatrix costDerivative(DoubleMatrix activations, DoubleMatrix output) {
        return activations.sub(output);
    }
}
