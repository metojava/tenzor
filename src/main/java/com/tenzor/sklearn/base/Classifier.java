package com.tenzor.sklearn.base;

import com.tenzor.core.Tensor;

public abstract class Classifier extends Estimator {
    public abstract Tensor predict(Tensor X);

    public abstract Tensor predictProba(Tensor X);

    public double score(Tensor X, Tensor y) {
        Tensor pred = predict(X);
        int correct = 0;
        int total = y.size();
        for (int i = 0; i < total; i++) {
            if (Math.round(pred.get(i)) == Math.round(y.get(i))) correct++;
        }
        return (double) correct / total;
    }
}
