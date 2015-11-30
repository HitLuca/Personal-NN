package DataImportExport;

import NeuralNetwork.DatasetElement;

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

    List<DatasetElement> dataset;
    List<DatasetElement> validation;

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
            if (b == true && k >= 50000) {
                validation.add(new DatasetElement());
            } else {
                dataset.add(new DatasetElement());
            }

            dataList = Arrays.asList(string.split(","));
            for (int i = 0; i < 10; i++) {
                if (b == true && k >= 50000) {
                    validation.get(k-50000).addOutput(Double.parseDouble(dataList.get(i)));
                } else {
                    dataset.get(k).addOutput(Double.parseDouble(dataList.get(i)));
                }
            }
            for (int i = 10; i < dataList.size(); i++) {
                if (b == true && k >= 50000) {
                    validation.get(k-50000).addInput(Double.parseDouble(dataList.get(i)));
                } else {
                    dataset.get(k).addInput(Double.parseDouble(dataList.get(i)));
                }
            }
            k++;
            string = reader.readLine();
        } while (string != null);
        reader.close();
    }

    public List<DatasetElement> getDataset() {
        return dataset;
    }

    public List<DatasetElement> getValidation() {
        return validation;
    }
}
