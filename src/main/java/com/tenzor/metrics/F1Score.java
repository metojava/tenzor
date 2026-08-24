package com.tenzor.metrics;

import com.tenzor.core.Tensor;

public class F1Score {
    public static double compute(Tensor yTrue, Tensor yPred) {
        return computeBinary(yTrue, yPred, 1.0);
    }

    public static double computeBinary(Tensor yTrue, Tensor yPred, double positiveClass) {
        double precision = Precision.computeBinary(yTrue, yPred, positiveClass);
        double recall = Recall.computeBinary(yTrue, yPred, positiveClass);
        return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    }
}
