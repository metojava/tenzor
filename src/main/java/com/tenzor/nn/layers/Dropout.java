package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Dropout extends Layer {
    private double rate;
    private boolean training = true;
    private Random rng = new Random();
    private boolean[] mask;

    public Dropout(double rate) {
        super("dropout_" + rate);
        this.rate = rate;
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        if (!training || rate == 0) {
            this.output = input;
            return output;
        }
        double scale = 1.0 / (1.0 - rate);
        double[] data = input.getData();
        double[] result = new double[data.length];
        mask = new boolean[data.length];
        for (int i = 0; i < data.length; i++) {
            if (rng.nextDouble() >= rate) {
                result[i] = data[i] * scale;
                mask[i] = true;
            } else {
                result[i] = 0;
                mask[i] = false;
            }
        }
        this.output = new Tensor(result, input.shape());
        return output;
    }

    public Tensor backward(Tensor gradOutput) {
        if (!training || rate == 0) return gradOutput;
        double scale = 1.0 / (1.0 - rate);
        double[] grad = gradOutput.getData();
        double[] result = new double[grad.length];
        for (int i = 0; i < grad.length; i++) {
            result[i] = mask[i] ? grad[i] * scale : 0;
        }
        return new Tensor(result, gradOutput.shape());
    }

    public void setTraining(boolean training) { this.training = training; }

    @Override
    public List<Variable> getParameters() {
        return new ArrayList<>();
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
