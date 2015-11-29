package NeuralNetwork;

import javax.activation.DataSource;
import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by hitluca on 29/11/15.
 */
public class DatasetElement {
    private List<Double> inputs;
    private List<Double> outputs;

    public DatasetElement() {
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
    }

    public DatasetElement(List<Double> inputs, List<Double> outputs) {
        inputs = new ArrayList<>(inputs);
        outputs = new ArrayList<>(outputs);
    }

    public List<Double> getInputs() {
        return inputs;
    }

    public void addInput(double d) {
        inputs.add(d);
    }

    public List<Double> getOutputs() {
        return outputs;
    }

    public void addOutput(double d) {
        outputs.add(d);
    }
}
