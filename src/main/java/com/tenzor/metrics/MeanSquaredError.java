package com.tenzor.metrics;

import com.tenzor.core.Tensor;

public class MeanSquaredError {
    public static double compute(Tensor yTrue, Tensor yPred) {
        return yTrue.sub(yPred).pow(2).mean().scalar();
    }
}
