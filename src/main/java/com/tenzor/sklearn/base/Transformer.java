package com.tenzor.sklearn.base;

import com.tenzor.core.Tensor;

public abstract class Transformer extends Estimator {
    public abstract Tensor fitTransform(Tensor X);

    public abstract Tensor transform(Tensor X);

    public abstract Tensor inverseTransform(Tensor X);
}
