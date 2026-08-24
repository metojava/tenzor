package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.core.TensorMath;
import com.tenzor.core.Shape;
import java.util.ArrayList;
import java.util.List;

public class LSTM extends Layer {
    private int units;
    private boolean returnSequences;
    private Variable kernel;
    private Variable recurrentKernel;
    private Variable bias;
    private Tensor lastOutput;
    private Tensor lastState;
    private Tensor lastCellState;

    public LSTM(int units) {
        this(units, false);
    }

    public LSTM(int units, boolean returnSequences) {
        super("lstm_" + units);
        this.units = units;
        this.returnSequences = returnSequences;
    }

    private void initWeights(int inputSize) {
        if (kernel != null) return;
        kernel = new Variable(TensorMath.xavierUniform(inputSize, 4 * units));
        kernel.setName(name + "_kernel");
        recurrentKernel = new Variable(TensorMath.xavierUniform(units, 4 * units));
        recurrentKernel.setName(name + "_recurrent_kernel");
        bias = Variable.zeros(4 * units);
        bias.setName(name + "_bias");
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        int[] dims = input.shape().dims();
        int batchSize = dims[0];
        int seqLen = dims[1];
        int inputSize = dims[2];

        initWeights(inputSize);

        double[] kData = kernel.getData();
        double[] rkData = recurrentKernel.getData();
        double[] bData = bias.getData();
        double[] inData = input.getData();

        double[] hState = new double[batchSize * units];
        double[] cState = new double[batchSize * units];
        double[] allOutputs = returnSequences ? new double[batchSize * seqLen * units] : null;

        for (int t = 0; t < seqLen; t++) {
            double[] gates = new double[batchSize * 4 * units];
            for (int n = 0; n < batchSize; n++) {
                for (int g = 0; g < 4 * units; g++) {
                    double sum = bData[g];
                    for (int i = 0; i < inputSize; i++) {
                        sum += inData[(n * seqLen + t) * inputSize + i] * kData[i * (4 * units) + g];
                    }
                    for (int u = 0; u < units; u++) {
                        sum += hState[n * units + u] * rkData[u * (4 * units) + g];
                    }
                    gates[n * (4 * units) + g] = sum;
                }
            }

            for (int n = 0; n < batchSize; n++) {
                for (int u = 0; u < units; u++) {
                    int iGate = n * (4 * units) + u;
                    int fGate = n * (4 * units) + u + units;
                    int cGate = n * (4 * units) + u + 2 * units;
                    int oGate = n * (4 * units) + u + 3 * units;

                    double inputGate = sigmoid(gates[iGate]);
                    double forgetGate = sigmoid(gates[fGate]);
                    double cellGate = Math.tanh(gates[cGate]);
                    double outputGate = sigmoid(gates[oGate]);

                    cState[n * units + u] = forgetGate * cState[n * units + u] + inputGate * cellGate;
                    hState[n * units + u] = outputGate * Math.tanh(cState[n * units + u]);
                }
            }

            if (returnSequences) {
                for (int n = 0; n < batchSize; n++) {
                    System.arraycopy(hState, n * units, allOutputs, (t * batchSize + n) * units, units);
                }
            }
        }

        lastState = new Tensor(hState, new Shape(batchSize, units));
        lastCellState = new Tensor(cState, new Shape(batchSize, units));
        if (returnSequences) {
            this.output = new Tensor(allOutputs, new Shape(batchSize, seqLen, units));
        } else {
            this.output = lastState;
        }
        return output;
    }

    private double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }

    @Override
    public void resetState() {
        lastState = null;
        lastCellState = null;
    }

    @Override
    public List<Variable> getParameters() {
        List<Variable> params = new ArrayList<>();
        params.add(kernel);
        params.add(recurrentKernel);
        params.add(bias);
        return params;
    }

    @Override
    public List<Tensor> getGradients() {
        return new ArrayList<>();
    }

    @Override
    public int[] getOutputShape(int[] inputShape) {
        if (returnSequences) return new int[]{inputShape[0], inputShape[1], units};
        return new int[]{inputShape[0], units};
    }
}
