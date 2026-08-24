package com.tenzor.nn.losses;

import com.tenzor.core.Tensor;

public class MeanAbsoluteError extends Loss {
    public MeanAbsoluteError() { super("mae"); }

    @Override
    public Tensor forward(Tensor predictions, Tensor targets) {
        return predictions.sub(targets).abs().mean();
    }

    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        Tensor diff = predictions.sub(targets);
        int n = predictions.size();
        return diff.elementWiseScalar(d -> (d > 0 ? 1.0 : (d < 0 ? -1.0 : 0.0)) / n);
    }
}
