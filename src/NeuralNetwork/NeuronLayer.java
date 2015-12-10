package NeuralNetwork;

import NeuralNetwork.Activations.ActivationFunction;
import NeuralNetwork.CostFunctions.CostFunction;
import org.jblas.DoubleMatrix;
import org.jblas.benchmark.Main;

/**
 * Created by hitluca on 09/12/15.
 */
public class NeuronLayer {
    private DoubleMatrix weights;
    private DoubleMatrix activations;
    private DoubleMatrix totals;
    private DoubleMatrix biases;

    private int neuronsNumber;

    private NeuronLayer previousLayer;
    private NeuronLayer nextLayer;

    private DoubleMatrix miniBatchBDeltas;
    private DoubleMatrix totalWDeltas;
    private DoubleMatrix totalBDeltas;

    private ActivationFunction activation;

    public NeuronLayer (int neurons, NeuronLayer previousLayer, boolean isHidden, ActivationFunction activation) {
        activations = new DoubleMatrix(neurons);
        neuronsNumber = neurons;
        this.activation = activation;

        if (isHidden) {
            totals = new DoubleMatrix(neurons);
            this.previousLayer = previousLayer;

            weights = new DoubleMatrix().randn(neurons, previousLayer.getNeuronsNumber()).mul(1/Math.sqrt(previousLayer.getNeuronsNumber()));
            biases = new DoubleMatrix().randn(neurons);

            miniBatchBDeltas = new DoubleMatrix().zeros(biases.rows);
            totalBDeltas = new DoubleMatrix().zeros(biases.rows);
            totalWDeltas = new DoubleMatrix().zeros(weights.rows, weights.columns);
        }
    }

    public void setNextLayer(NeuronLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public void processInput(boolean dropout, boolean test) {
        totals = (weights.mmul(previousLayer.getActivations())).add(biases);
        if (!test) {
            activations = activation.apply(totals);
            if (dropout) {
                for (int i=0; i<activations.rows; i++) {
                    if (Math.random()<0.5) {
                        activations.put(i, 0);
                    }
                }
            }
        } else {
            if (dropout) {
                totals.muli(0.5);
            }
            activations = activation.apply(totals);
        }
    }

    public void calculateLastLayerBWDeltas(CostFunction costFunction, DoubleMatrix output) {
        miniBatchBDeltas = costFunction.costDerivative(activations, output, totals, activation);
        totalBDeltas.addi(miniBatchBDeltas);
        totalWDeltas.addi(miniBatchBDeltas.mmul(previousLayer.getActivations().transpose()));
    }

    public void calculateBWDeltas() {
        miniBatchBDeltas = ((nextLayer.getWeights().transpose()).mmul(nextLayer.getMiniBatchBDeltas())).mul(activation.primeDerivative(totals));
        totalBDeltas.addi(miniBatchBDeltas);
        totalWDeltas.addi(miniBatchBDeltas.mmul(previousLayer.getActivations().transpose()));
    }

    public void updateWeightsBiases(double learningRate, double regularization, int n, int miniBatchSize) {
        weights = (weights.mul(1.0 - (learningRate * regularization / n))).sub(totalWDeltas.mul(learningRate / miniBatchSize));
        biases.subi(totalBDeltas.mul(learningRate / miniBatchSize));

        totalBDeltas = new DoubleMatrix().zeros(biases.rows);
        miniBatchBDeltas = new DoubleMatrix().zeros(biases.rows);
        totalWDeltas = new DoubleMatrix().zeros(weights.rows, weights.columns);
    }

    public void setActivations(DoubleMatrix activations) {
        this.activations = activations;
    }

    public DoubleMatrix getMiniBatchBDeltas() {
        return miniBatchBDeltas;
    }

    public DoubleMatrix calculateBackwardMatrix() {
        totals = activation.inverse(activations).sub(biases);
        return (weights.transpose()).mmul(totals);
    }

    public DoubleMatrix getActivations() {
        return activations;
    }

    public Double getSumWeights() {
        double sumOfSquares = 0;
        for (int i=0; i<weights.rows; i++) {
            for (int j=0; j<weights.columns; j++) {
                sumOfSquares += Math.pow(weights.get(i, j), 2);
            }
        }
        return sumOfSquares;
    }

    public int getNeuronsNumber() {
        return neuronsNumber;
    }

    public DoubleMatrix getWeights() {
        return weights;
    }
}
