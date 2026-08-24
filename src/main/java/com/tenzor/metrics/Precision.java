package com.tenzor.metrics;

import com.tenzor.core.Tensor;

public class Precision {
    public static double compute(Tensor yTrue, Tensor yPred) {
        return computeBinary(yTrue, yPred, 1.0);
    }

    public static double computeBinary(Tensor yTrue, Tensor yPred, double positiveClass) {
        double tp = 0, fp = 0;
        int n = yTrue.size();
        for (int i = 0; i < n; i++) {
            boolean predPos = Math.round(yPred.get(i)) == positiveClass;
            boolean truePos = Math.round(yTrue.get(i)) == positiveClass;
            if (predPos && truePos) tp++;
            else if (predPos && !truePos) fp++;
        }
        return tp + fp > 0 ? tp / (tp + fp) : 0;
    }
}
