package Miscellaneous;

import javafx.util.Pair;
import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by hitluca on 08/10/15.
 */
public class CSVLoader {
    BufferedReader reader;

    List<Pair<DoubleMatrix, DoubleMatrix>> dataset;
    List<Pair<DoubleMatrix, DoubleMatrix>> validation;

    public CSVLoader(String filename) throws IOException {
        try {
            reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void importData(boolean b) throws IOException {
        dataset = new ArrayList<>();
        validation = new ArrayList<>();

        String string;
        List<String> dataList;

        int k = 0;

        string = reader.readLine();
        do {
            dataList = Arrays.asList(string.split(","));

            DoubleMatrix output = new DoubleMatrix(10);
            for (int i = 0; i < 10; i++) {
                output.put(i, Double.parseDouble(dataList.get(i)));
            }
            DoubleMatrix input = new DoubleMatrix(dataList.size()-10);
            for (int i = 10; i < dataList.size(); i++) {
                input.put(i-10, Double.parseDouble(dataList.get(i)));
            }

            Pair<DoubleMatrix, DoubleMatrix> p = new Pair(input, output);

            if (b==true && k>=50000) {
                validation.add(p);
            }
            else
            {
                dataset.add(p);
            }
            k++;
            string = reader.readLine();
        } while (string != null);
        reader.close();
    }

    public List<Pair<DoubleMatrix, DoubleMatrix>> getDataset() {
        return dataset;
    }

    public List<Pair<DoubleMatrix, DoubleMatrix>> getValidation() {
        return validation;
    }
}
