package com.tenzor.sklearn.preprocessing;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;

public class MinMaxScaler extends Transformer {
    private double min;
    private double max;
    private Tensor dataMin;
    private Tensor dataMax;
    private Tensor scale;

    public MinMaxScaler() { this(0, 1); }

    public MinMaxScaler(double min, double max) {
        super();
        this.min = min;
        this.max = max;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        dataMin = X.min(0);
        dataMax = X.max(0);
        scale = dataMax.sub(dataMin);
        double[] s = scale.getData();
        for (int i = 0; i < s.length; i++) {
            if (s[i] < 1e-10) s[i] = 1.0;
        }
        scale = new Tensor(s, scale.shape());
        fitted = true;
    }

    @Override
    public Tensor transform(Tensor X) {
        if (!fitted) throw new IllegalStateException("MinMaxScaler not fitted");
        Tensor result = X.sub(dataMin).div(scale);
        double range = max - min;
        if (Math.abs(range - 1.0) > 1e-10) {
            result = result.mul(range).add(min);
        }
        return result;
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        fit(X, null);
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) {
        if (!fitted) throw new IllegalStateException("MinMaxScaler not fitted");
        double range = max - min;
        Tensor result = X;
        if (Math.abs(range - 1.0) > 1e-10) {
            result = result.sub(min).div(range);
        }
        return result.mul(scale).add(dataMin);
    }
}
