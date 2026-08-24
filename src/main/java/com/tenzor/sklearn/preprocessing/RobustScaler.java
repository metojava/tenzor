package com.tenzor.sklearn.preprocessing;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;

public class RobustScaler extends Transformer {
    private Tensor median;
    private Tensor iqr;

    public RobustScaler() { super(); }

    @Override
    public void fit(Tensor X, Tensor y) {
        int m = X.shape().dim(X.shape().numDims() - 1);
        double[][] xData = X.to2D();
        median = new Tensor(new double[m], new Shape(m));
        double[] q1 = new double[m];
        double[] q3 = new double[m];

        for (int j = 0; j < m; j++) {
            double[] col = new double[X.shape().dim(0)];
            for (int i = 0; i < col.length; i++) col[i] = xData[i][j];
            java.util.Arrays.sort(col);
            median.getData()[j] = col[col.length / 2];
            q1[j] = col[col.length / 4];
            q3[j] = col[3 * col.length / 4];
        }

        iqr = new Tensor(new double[m], new Shape(m));
        for (int j = 0; j < m; j++) {
            double val = q3[j] - q1[j];
            iqr.getData()[j] = val < 1e-10 ? 1.0 : val;
        }
        fitted = true;
    }

    @Override
    public Tensor transform(Tensor X) {
        if (!fitted) throw new IllegalStateException("RobustScaler not fitted");
        return X.sub(median).div(iqr);
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        fit(X, null);
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) {
        if (!fitted) throw new IllegalStateException("RobustScaler not fitted");
        return X.mul(iqr).add(median);
    }
}
