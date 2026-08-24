package com.tenzor.sklearn.svm;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Regressor;

public class SVR extends Regressor {
    private double C;
    private double epsilon;
    private double learningRate;
    private int maxIter;
    private Tensor weights;
    private Tensor bias;

    public SVR() { this(1.0, 0.1, 0.001, 1000); }

    public SVR(double C, double epsilon, double learningRate, int maxIter) {
        super();
        this.C = C;
        this.epsilon = epsilon;
        this.learningRate = learningRate;
        this.maxIter = maxIter;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        weights = Tensor.zeros(m);
        bias = Tensor.zeros(1);
        double[] yData = y.getData();
        double[][] xData = X.to2D();

        for (int iter = 0; iter < maxIter; iter++) {
            for (int i = 0; i < n; i++) {
                double pred = 0;
                for (int j = 0; j < m; j++) pred += weights.get(j) * xData[i][j];
                pred += bias.get(0);

                double error = pred - yData[i];
                double[] w = weights.getData();
                double[] b = bias.getData();

                if (Math.abs(error) > epsilon) {
                    double sign = error > 0 ? 1 : -1;
                    for (int j = 0; j < m; j++) {
                        w[j] -= learningRate * (w[j] + C * sign * xData[i][j]);
                    }
                    b[0] -= learningRate * (C * sign);
                } else {
                    for (int j = 0; j < m; j++) w[j] -= learningRate * w[j];
                }
            }
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        double[][] xData = X.to2D();
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            double pred = 0;
            for (int j = 0; j < m; j++) pred += weights.get(j) * xData[i][j];
            result[i] = pred + bias.get(0);
        }
        return new Tensor(result, new Shape(n));
    }
}
