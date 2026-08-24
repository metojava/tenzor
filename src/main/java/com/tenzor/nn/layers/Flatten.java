package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import java.util.ArrayList;
import java.util.List;

public class Flatten extends Layer {
    public Flatten() {
        super("flatten");
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        int[] dims = input.shape().dims();
        int batchSize = dims[0];
        int flatSize = 1;
        for (int i = 1; i < dims.length; i++) flatSize *= dims[i];
        this.output = input.reshape(batchSize, flatSize);
        return this.output;
    }

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
        int flatSize = 1;
        for (int i = 1; i < inputShape.length; i++) flatSize *= inputShape[i];
        return new int[]{inputShape[0], flatSize};
    }
}
