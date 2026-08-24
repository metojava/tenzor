package com.tenzor.metrics;

import com.tenzor.core.Tensor;

public class R2Score {
    public static double compute(Tensor yTrue, Tensor yPred) {
        Tensor diff = yTrue.sub(yPred);
        double ssRes = diff.pow(2).sum().scalar();
        double yMean = yTrue.mean().scalar();
        double ssTot = yTrue.sub(yMean).pow(2).sum().scalar();
        return 1.0 - ssRes / ssTot;
    }
}
