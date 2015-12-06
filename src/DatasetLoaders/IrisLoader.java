package DatasetLoaders;

import javafx.util.Pair;
import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Created by hitluca on 06/12/15.
 */
public class IrisLoader implements Loader {
    BufferedReader reader;

    Map<String, Integer> nameMap;
    List<Pair<DoubleMatrix, DoubleMatrix>> container;
    List<Pair<DoubleMatrix, DoubleMatrix>> train;
    List<Pair<DoubleMatrix, DoubleMatrix>> validation;
    List<Pair<DoubleMatrix, DoubleMatrix>> test;

    public IrisLoader(String filename) throws IOException {
        try {
            reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void importData(boolean b) throws IOException {
        train = new ArrayList<>();
        validation = new ArrayList<>();
        test = new ArrayList<>();
        container = new ArrayList<>();


        String string;
        List<String> dataList;

        string = reader.readLine();
        do {
            dataList = Arrays.asList(string.split(","));

            DoubleMatrix input = new DoubleMatrix(4);
            DoubleMatrix output = new DoubleMatrix().zeros(3);

            for (int i = 0; i < input.rows; i++) {
                input.put(i, Double.parseDouble(dataList.get(i)));
            }
            String plantType = dataList.get(input.rows);
            switch (plantType) {
                case "Iris-setosa": {
                    output.put(0, 1.0);
                    break;
                }
                case "Iris-versicolor": {
                    output.put(1, 1.0);
                    break;
                }
                case "Iris-virginica": {
                    output.put(2, 1.0);
                    break;
                }
            }

            Pair<DoubleMatrix, DoubleMatrix> p = new Pair(input, output);
            container.add(p);
            string = reader.readLine();
        } while (string != null && !string.equals(""));
        reader.close();


        Collections.shuffle(container);
        train.addAll(container.subList(0, 104));
        validation.addAll(container.subList(105, 127));
        test.addAll(container.subList(128, 149));
    }

    public List<Pair<DoubleMatrix, DoubleMatrix>> getTrain() {
        return train;
    }

    public List<Pair<DoubleMatrix, DoubleMatrix>> getValidation() {
        return validation;
    }

    public List<Pair<DoubleMatrix, DoubleMatrix>> getTest() {
        return test;
    }
}
