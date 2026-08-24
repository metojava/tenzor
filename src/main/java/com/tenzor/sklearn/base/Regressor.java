package com.tenzor.sklearn.base;

import com.tenzor.core.Tensor;

public abstract class Regressor extends Estimator {
    public abstract Tensor predict(Tensor X);

    public double score(Tensor X, Tensor y) {
        Tensor pred = predict(X);
        Tensor diff = y.sub(pred);
        Tensor ssRes = diff.pow(2).sum();
        Tensor yMean = y.mean();
        Tensor ssTot = y.sub(yMean).pow(2).sum();
        return 1.0 - ssRes.div(ssTot).scalar();
    }
}
