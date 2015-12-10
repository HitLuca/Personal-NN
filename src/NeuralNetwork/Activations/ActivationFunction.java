package NeuralNetwork.Activations;

import org.jblas.DoubleMatrix;

/**
 * Created by hitluca on 10/12/15.
 */
public interface ActivationFunction {
    DoubleMatrix apply(DoubleMatrix input);

    DoubleMatrix primeDerivative(DoubleMatrix input);

    DoubleMatrix inverse(DoubleMatrix input);
}
