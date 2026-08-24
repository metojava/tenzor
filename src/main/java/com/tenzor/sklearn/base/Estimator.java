package com.tenzor.sklearn.base;

import com.tenzor.core.Tensor;

public abstract class Estimator {
    protected boolean fitted = false;
    protected String name;

    public Estimator() { this.name = getClass().getSimpleName(); }

    public abstract void fit(Tensor X, Tensor y);

    public boolean isFitted() { return fitted; }

    public String getName() { return name; }
}
