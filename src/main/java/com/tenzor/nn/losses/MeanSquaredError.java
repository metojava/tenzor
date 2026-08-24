package com.tenzor.nn.losses;

import com.tenzor.core.Tensor;

public class MeanSquaredError extends Loss {
    public MeanSquaredError() { super("mse"); }

    @Override
    public Tensor forward(Tensor predictions, Tensor targets) {
        Tensor diff = predictions.sub(targets);
        Tensor squared = diff.pow(2);
        return squared.mean();
    }

    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        Tensor diff = predictions.sub(targets);
        int n = predictions.size();
        return diff.mul(2.0 / n);
    }
}
