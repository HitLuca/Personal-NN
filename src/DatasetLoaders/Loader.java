package DatasetLoaders;

import javafx.util.Pair;
import org.jblas.DoubleMatrix;

import java.io.IOException;
import java.util.List;

/**
 * Created by hitluca on 06/12/15.
 */
public interface Loader {
    void importData(boolean b) throws IOException;

    List<Pair<DoubleMatrix, DoubleMatrix>> getTrain();

    List<Pair<DoubleMatrix, DoubleMatrix>> getValidation();

    List<Pair<DoubleMatrix, DoubleMatrix>> getTest();
}
