package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.core.Shape;
import java.util.ArrayList;
import java.util.List;

public class BatchNormalization extends Layer {
    private int axis;
    private double momentum;
    private double epsilon;
    private Variable gamma;
    private Variable beta;
    private Tensor runningMean;
    private Tensor runningVar;
    private Tensor normalizedInput;
    private Tensor batchMean;
    private Tensor batchVar;
    private boolean training = true;

    public BatchNormalization() {
        this(-1, 0.99, 1e-5);
    }

    public BatchNormalization(int axis, double momentum, double epsilon) {
        super("batchnorm");
        this.axis = axis;
        this.momentum = momentum;
        this.epsilon = epsilon;
    }

    private void initParams(int channels) {
        if (gamma != null) return;
        gamma = Variable.ones(channels);
        gamma.setName(name + "_gamma");
        beta = Variable.zeros(channels);
        beta.setName(name + "_beta");
        runningMean = Tensor.zeros(channels);
        runningVar = Tensor.ones(channels);
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        int[] dims = input.shape().dims();
        int channels = dims[dims.length - 1];
        initParams(channels);

        int batchSize = dims[0];
        int spatialSize = 1;
        for (int i = 1; i < dims.length - 1; i++) spatialSize *= dims[i];
        int total = batchSize * spatialSize;

        double[] inData = input.getData();
        double[] mean = new double[channels];
        double[] var = new double[channels];

        for (int c = 0; c < channels; c++) {
            double sum = 0;
            for (int n = 0; n < batchSize; n++) {
                for (int s = 0; s < spatialSize; s++) {
                    int idx = ((n * spatialSize + s) * channels) + c;
                    sum += inData[idx];
                }
            }
            mean[c] = sum / total;
        }

        for (int c = 0; c < channels; c++) {
            double sum = 0;
            for (int n = 0; n < batchSize; n++) {
                for (int s = 0; s < spatialSize; s++) {
                    int idx = ((n * spatialSize + s) * channels) + c;
                    double diff = inData[idx] - mean[c];
                    sum += diff * diff;
                }
            }
            var[c] = sum / total;
        }

        batchMean = new Tensor(mean, new Shape(channels));
        batchVar = new Tensor(var, new Shape(channels));

        if (training) {
            runningMean = runningMean.mul(momentum).add(batchMean.mul(1 - momentum));
            runningVar = runningVar.mul(momentum).add(batchVar.mul(1 - momentum));
        }

        Tensor centered = input;
        double[] gData = gamma.getData();
        double[] bData = beta.getData();
        double[] outData = new double[inData.length];
        double[] m = training ? mean : runningMean.getData();
        double[] v = training ? var : runningVar.getData();

        for (int idx = 0; idx < inData.length; idx++) {
            int c = idx % channels;
            outData[idx] = gData[c] * (inData[idx] - m[c]) / Math.sqrt(v[c] + epsilon) + bData[c];
        }

        this.normalizedInput = new Tensor(outData, input.shape());
        this.output = this.normalizedInput;
        return output;
    }

    public void setTraining(boolean training) { this.training = training; }

    @Override
    public List<Variable> getParameters() {
        List<Variable> params = new ArrayList<>();
        params.add(gamma);
        params.add(beta);
        return params;
    }

    @Override
    public List<Tensor> getGradients() {
        return new ArrayList<>();
    }

    @Override
    public int[] getOutputShape(int[] inputShape) {
        return inputShape.clone();
    }
}
