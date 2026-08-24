package com.tenzor.sklearn.preprocessing;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;

public class SimpleImputer extends Transformer {
    private String strategy;
    private double fillValue;
    private Tensor statistics;

    public SimpleImputer() { this("mean", 0); }

    public SimpleImputer(String strategy, double fillValue) {
        super();
        this.strategy = strategy;
        this.fillValue = fillValue;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int m = X.shape().dim(X.shape().numDims() - 1);
        double[][] xData = X.to2D();
        double[] stats = new double[m];

        for (int j = 0; j < m; j++) {
            double sum = 0;
            int count = 0;
            for (int i = 0; i < X.shape().dim(0); i++) {
                if (!Double.isNaN(xData[i][j])) {
                    sum += xData[i][j];
                    count++;
                }
            }
            switch (strategy) {
                case "mean": stats[j] = count > 0 ? sum / count : fillValue; break;
                case "median":
                    double[] vals = new double[count];
                    int idx = 0;
                    for (int i = 0; i < X.shape().dim(0); i++) {
                        if (!Double.isNaN(xData[i][j])) vals[idx++] = xData[i][j];
                    }
                    java.util.Arrays.sort(vals);
                    stats[j] = vals.length > 0 ? vals[vals.length / 2] : fillValue;
                    break;
                case "constant": stats[j] = fillValue; break;
                default: stats[j] = fillValue;
            }
        }
        statistics = new Tensor(stats, new Shape(m));
        fitted = true;
    }

    @Override
    public Tensor transform(Tensor X) {
        if (!fitted) throw new IllegalStateException("SimpleImputer not fitted");
        double[] result = X.getData().clone();
        int m = X.shape().dim(X.shape().numDims() - 1);
        for (int i = 0; i < X.shape().dim(0); i++) {
            for (int j = 0; j < m; j++) {
                if (Double.isNaN(result[i * m + j])) {
                    result[i * m + j] = statistics.get(j);
                }
            }
        }
        return new Tensor(result, X.shape());
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        fit(X, null);
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) { return X; }
}
