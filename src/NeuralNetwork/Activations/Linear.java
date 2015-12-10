package NeuralNetwork.Activations;

import org.jblas.DoubleMatrix;

/**
 * Created by hitluca on 10/12/15.
 */
public class Linear implements ActivationFunction {
    public DoubleMatrix apply(DoubleMatrix input) {
        return input;
    }

    public DoubleMatrix primeDerivative(DoubleMatrix input) {
        return new DoubleMatrix().ones(input.rows);
    }

    public DoubleMatrix inverse(DoubleMatrix input) {
        //TODO:implement
        return null;
    }
}
