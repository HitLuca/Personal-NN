package NeuralNetwork.Activations;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by hitluca on 10/12/15.
 */
public class TanH implements ActivationFunction {
    public DoubleMatrix apply(DoubleMatrix input) {
        return MatrixFunctions.tanh(input);
    }

    public DoubleMatrix primeDerivative(DoubleMatrix input) {
        return (new DoubleMatrix().ones(input.rows)).sub(MatrixFunctions.pow(apply(input), 2));
    }

    public DoubleMatrix inverse(DoubleMatrix input) {
        //TODO:implement
        return null;
    }
}
