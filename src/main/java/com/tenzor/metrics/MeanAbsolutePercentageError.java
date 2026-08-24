package com.tenzor.metrics;

import com.tenzor.core.Tensor;

public class MeanAbsolutePercentageError {
    public static double compute(Tensor yTrue, Tensor yPred) {
        double[] t = yTrue.getData();
        double[] p = yPred.getData();
        double sum = 0;
        for (int i = 0; i < t.length; i++) {
            double absT = Math.abs(t[i]);
            if (absT > 1e-10) {
                sum += Math.abs((t[i] - p[i]) / t[i]);
            }
        }
        return 100.0 * sum / t.length;
    }
}
