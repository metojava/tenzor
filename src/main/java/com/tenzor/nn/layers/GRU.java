package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.core.TensorMath;
import com.tenzor.core.Shape;
import java.util.ArrayList;
import java.util.List;

public class GRU extends Layer {
    private int units;
    private boolean returnSequences;
    private Variable kernel;
    private Variable recurrentKernel;
    private Variable bias;
    private Tensor lastState;

    public GRU(int units) {
        this(units, false);
    }

    public GRU(int units, boolean returnSequences) {
        super("gru_" + units);
        this.units = units;
        this.returnSequences = returnSequences;
    }

    private void initWeights(int inputSize) {
        if (kernel != null) return;
        kernel = new Variable(TensorMath.xavierUniform(inputSize, 3 * units));
        kernel.setName(name + "_kernel");
        recurrentKernel = new Variable(TensorMath.xavierUniform(units, 3 * units));
        recurrentKernel.setName(name + "_recurrent_kernel");
        bias = Variable.zeros(3 * units);
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
        double[] allOutputs = returnSequences ? new double[batchSize * seqLen * units] : null;

        for (int t = 0; t < seqLen; t++) {
            double[] z = new double[batchSize * units];
            double[] r = new double[batchSize * units];
            double[] n = new double[batchSize * units];

            for (int nIdx = 0; nIdx < batchSize; nIdx++) {
                for (int u = 0; u < units; u++) {
                    double zSum = bData[u];
                    double rSum = bData[u + units];
                    double nSum = bData[u + 2 * units];

                    for (int i = 0; i < inputSize; i++) {
                        double x = inData[(nIdx * seqLen + t) * inputSize + i];
                        zSum += x * kData[i * (3 * units) + u];
                        rSum += x * kData[i * (3 * units) + u + units];
                        nSum += x * kData[i * (3 * units) + u + 2 * units];
                    }
                    for (int u2 = 0; u2 < units; u2++) {
                        double h = hState[nIdx * units + u2];
                        zSum += h * rkData[u2 * (3 * units) + u];
                        rSum += h * rkData[u2 * (3 * units) + u + units];
                    }

                    z[nIdx * units + u] = sigmoid(zSum);
                    r[nIdx * units + u] = sigmoid(rSum);

                    double nSum2 = nSum;
                    for (int u2 = 0; u2 < units; u2++) {
                        nSum2 += r[nIdx * units + u2] * hState[nIdx * units + u2] * rkData[u2 * (3 * units) + u + 2 * units];
                    }
                    n[nIdx * units + u] = Math.tanh(nSum2);

                    hState[nIdx * units + u] = (1 - z[nIdx * units + u]) * n[nIdx * units + u] + z[nIdx * units + u] * hState[nIdx * units + u];
                }
            }

            if (returnSequences) {
                for (int nIdx = 0; nIdx < batchSize; nIdx++) {
                    System.arraycopy(hState, nIdx * units, allOutputs, (t * batchSize + nIdx) * units, units);
                }
            }
        }

        lastState = new Tensor(hState, new Shape(batchSize, units));
        if (returnSequences) {
            this.output = new Tensor(allOutputs, new Shape(batchSize, seqLen, units));
        } else {
            this.output = lastState;
        }
        return output;
    }

    private double sigmoid(double x) { return 1.0 / (1.0 + Math.exp(-x)); }

    @Override
    public void resetState() { lastState = null; }

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
