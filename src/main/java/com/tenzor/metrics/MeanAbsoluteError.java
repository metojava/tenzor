package com.tenzor.metrics;

import com.tenzor.core.Tensor;

public class MeanAbsoluteError {
    public static double compute(Tensor yTrue, Tensor yPred) {
        return yTrue.sub(yPred).abs().mean().scalar();
    }
}
