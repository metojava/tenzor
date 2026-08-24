package com.tenzor.nn.losses;

import com.tenzor.core.Tensor;

public class Hinge extends Loss {
    public Hinge() { super("hinge"); }

    @Override
    public Tensor forward(Tensor predictions, Tensor targets) {
        Tensor one = Tensor.ones(predictions.shape());
        Tensor loss = one.sub(targets.mul(predictions)).elementWiseScalar(d -> Math.max(0, d));
        return loss.mean();
    }

    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        int n = predictions.size();
        Tensor margin = targets.mul(predictions).neg().add(1.0);
        return targets.mul(
            margin.elementWiseScalar(d -> d > 0 ? -1.0 / n : 0.0));
    }
}
