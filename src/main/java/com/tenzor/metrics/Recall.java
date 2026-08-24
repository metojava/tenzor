package com.tenzor.metrics;

import com.tenzor.core.Tensor;

public class Recall {
    public static double compute(Tensor yTrue, Tensor yPred) {
        return computeBinary(yTrue, yPred, 1.0);
    }

    public static double computeBinary(Tensor yTrue, Tensor yPred, double positiveClass) {
        double tp = 0, fn = 0;
        int n = yTrue.size();
        for (int i = 0; i < n; i++) {
            boolean truePos = Math.round(yTrue.get(i)) == positiveClass;
            boolean predPos = Math.round(yPred.get(i)) == positiveClass;
            if (truePos && predPos) tp++;
            else if (truePos && !predPos) fn++;
        }
        return tp + fn > 0 ? tp / (tp + fn) : 0;
    }
}
