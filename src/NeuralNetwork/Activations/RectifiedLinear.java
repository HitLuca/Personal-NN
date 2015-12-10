package NeuralNetwork.Activations;

import org.jblas.DoubleMatrix;

/**
 * Created by hitluca on 10/12/15.
 */
public class RectifiedLinear implements ActivationFunction {
    public DoubleMatrix apply(DoubleMatrix input) {
        DoubleMatrix retified = input.dup();
        for (int i=0; i<retified.rows; i++) {
            if (retified.get(i)<0) {
                retified.put(i, 0);
            }
        }
        return retified;
    }

    public DoubleMatrix primeDerivative(DoubleMatrix input) {
        DoubleMatrix derivative = new DoubleMatrix().ones(input.rows);
        for (int i=0; i<derivative.rows; i++) {
            if (derivative.get(i)<0) {
                derivative.put(i, 1);
            }
        }
        return derivative;
    }

    public DoubleMatrix inverse(DoubleMatrix input) {
        //TODO:implement
        return null;
    }
}
