package Miscellaneous;

import org.jblas.DoubleMatrix;

/**
 * Created by hitluca on 05/12/15.
 */
public class NormalizerDenormalizer {
    public static double denormalize(double d, double min, double max) {
        return (d*(max - min) + min);
    }

    public static double normalize(double d, double min, double max) {
        return (d-min)/(max-min);
    }

    public static DoubleMatrix normalizeMatrix(DoubleMatrix matrix, double min, double max) {
        if (matrix.rows==1) {
            return matrix.put(0, 1.0);
        }

        DoubleMatrix result = new DoubleMatrix(matrix.rows);
        for (int i = 0; i < matrix.rows; i++) {
            result.put(i, normalize(matrix.get(i), min, max));
        }
        return result;
    }

    public static DoubleMatrix denormalizeMatrix(DoubleMatrix matrix, double min, double max) {
        if (matrix.rows==1) {
            return matrix.put(0, 1.0);
        }

        DoubleMatrix result = new DoubleMatrix(matrix.rows);
        for (int i = 0; i < matrix.rows; i++) {
            result.put(i, denormalize(matrix.get(i), min, max));
        }
        return result;
    }
}
