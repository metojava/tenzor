package com.tenzor.sklearn.preprocessing;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;

public class Normalizer extends Transformer {
    private String norm;

    public Normalizer() { this("l2"); }

    public Normalizer(String norm) {
        super();
        this.norm = norm;
    }

    @Override
    public void fit(Tensor X, Tensor y) { fitted = true; }

    @Override
    public Tensor transform(Tensor X) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        double[][] xData = X.to2D();
        double[][] result = new double[n][m];

        for (int i = 0; i < n; i++) {
            double normVal = 0;
            switch (norm) {
                case "l1":
                    for (int j = 0; j < m; j++) normVal += Math.abs(xData[i][j]);
                    break;
                case "l2":
                    for (int j = 0; j < m; j++) normVal += xData[i][j] * xData[i][j];
                    normVal = Math.sqrt(normVal);
                    break;
                case "max":
                    for (int j = 0; j < m; j++) normVal = Math.max(normVal, Math.abs(xData[i][j]));
                    break;
                default:
                    for (int j = 0; j < m; j++) normVal += xData[i][j] * xData[i][j];
                    normVal = Math.sqrt(normVal);
            }
            if (normVal < 1e-10) normVal = 1.0;
            for (int j = 0; j < m; j++) result[i][j] = xData[i][j] / normVal;
        }
        return new Tensor(result);
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) { return X; }
}
