package com.tenzor.nn.losses;

import com.tenzor.core.Tensor;

public class CategoricalCrossentropy extends Loss {
    private double epsilon;

    public CategoricalCrossentropy() { this(1e-7); }

    public CategoricalCrossentropy(double epsilon) {
        super("categorical_crossentropy");
        this.epsilon = epsilon;
    }

    @Override
    public Tensor forward(Tensor predictions, Tensor targets) {
        Tensor clipped = predictions.elementWiseScalar(
            d -> Math.max(epsilon, Math.min(1 - epsilon, d)));
        Tensor logPred = clipped.log();
        Tensor loss = targets.mul(logPred).neg().sum(1);
        return loss.mean();
    }

    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        Tensor clipped = predictions.elementWiseScalar(
            d -> Math.max(epsilon, Math.min(1 - epsilon, d)));
        int n = predictions.shape().dim(0);
        return clipped.sub(targets).div(n);
    }
}
