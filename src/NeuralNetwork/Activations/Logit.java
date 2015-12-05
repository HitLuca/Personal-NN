package NeuralNetwork.Activations;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by hitluca on 05/12/15.
 */
public class Logit {
    public static DoubleMatrix logit(DoubleMatrix input) {
        DoubleMatrix a = new DoubleMatrix().ones(input.rows).sub(input);
        return MatrixFunctions.log(input.div(a));
    }
}
