package com.tenzor.metrics;

import com.tenzor.core.Tensor;

public class Accuracy {
    public static double compute(Tensor yTrue, Tensor yPred) {
        int correct = 0;
        int n = yTrue.size();
        for (int i = 0; i < n; i++) {
            if (Math.round(yTrue.get(i)) == Math.round(yPred.get(i))) correct++;
        }
        return (double) correct / n;
    }
}
