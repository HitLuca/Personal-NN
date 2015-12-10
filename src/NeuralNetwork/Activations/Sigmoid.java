package NeuralNetwork.Activations;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by hitluca on 05/12/15.
 */
public class Sigmoid implements ActivationFunction {
    public DoubleMatrix apply(DoubleMatrix input) {
        DoubleMatrix Denom = (MatrixFunctions.exp(input.mul(-1))).addi(1);
        return Denom.rdivi(1);
    }

    public DoubleMatrix primeDerivative(DoubleMatrix input) {
        DoubleMatrix M = apply(input);
        return M.mul((M.mul(-1)).addi(1));
    }

    public DoubleMatrix inverse(DoubleMatrix input) {
        DoubleMatrix a = new DoubleMatrix().ones(input.rows).sub(input);
        return MatrixFunctions.log(input.div(a));
    }
}
