package NeuralNetwork.Activations;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by hitluca on 05/12/15.
 */
public class Sigmoid {
    public static DoubleMatrix sigmoid(DoubleMatrix input) {
        DoubleMatrix Denom = (MatrixFunctions.exp(input.mul(-1))).addi(1);
        return Denom.rdivi(1);
    }

    public static DoubleMatrix sigmoidPrime(DoubleMatrix input) {
        DoubleMatrix M = sigmoid(input);
        return M.mul((M.mul(-1)).addi(1));
    }
}
