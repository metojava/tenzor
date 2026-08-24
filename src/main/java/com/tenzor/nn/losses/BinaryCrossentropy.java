package com.tenzor.nn.losses;

import com.tenzor.core.Tensor;

public class BinaryCrossentropy extends Loss {
    private double epsilon;

    public BinaryCrossentropy() { this(1e-7); }

    public BinaryCrossentropy(double epsilon) {
        super("binary_crossentropy");
        this.epsilon = epsilon;
    }

    @Override
    public Tensor forward(Tensor predictions, Tensor targets) {
        Tensor clipped = predictions.elementWiseScalar(
            d -> Math.max(epsilon, Math.min(1 - epsilon, d)));
        Tensor oneMinus = clipped.neg().add(1.0).elementWiseScalar(
            d -> Math.max(epsilon, d));
        Tensor loss = targets.mul(clipped.log()).add(
            targets.neg().add(1.0).mul(oneMinus.log())).neg();
        return loss.mean();
    }

    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        Tensor clipped = predictions.elementWiseScalar(
            d -> Math.max(epsilon, Math.min(1 - epsilon, d)));
        int n = predictions.size();
        return clipped.sub(targets).div(n * 1.0).mul(
            clipped.neg().add(1.0).mul(clipped).elementWiseScalar(
                d -> d < epsilon ? 1.0 : 1.0 / d));
    }
}
