package com.tenzor.sklearn.linear_model;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Regressor;

public class ElasticNet extends Regressor {
    private double alpha;
    private double l1Ratio;
    private double learningRate;
    private int maxIter;
    private boolean fitIntercept;
    private Tensor coefficients;
    private Tensor intercept;

    public ElasticNet() { this(1.0, 0.5, true); }

    public ElasticNet(double alpha, double l1Ratio, boolean fitIntercept) {
        this(alpha, l1Ratio, fitIntercept, 0.01, 1000);
    }

    public ElasticNet(double alpha, double l1Ratio, boolean fitIntercept, double learningRate, int maxIter) {
        super();
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.fitIntercept = fitIntercept;
        this.learningRate = learningRate;
        this.maxIter = maxIter;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        coefficients = Tensor.zeros(m);
        intercept = Tensor.zeros(1);

        for (int iter = 0; iter < maxIter; iter++) {
            Tensor pred = predict(X);
            Tensor error = pred.sub(y);
            Tensor grad = X.transpose().matmul(error).div(n);
            double[] c = coefficients.getData();
            double[] g = grad.getData();
            for (int i = 0; i < m; i++) {
                double l1Penalty = alpha * l1Ratio * sign(c[i]);
                double l2Penalty = alpha * (1 - l1Ratio) * 2 * c[i];
                c[i] -= learningRate * (g[i] + l1Penalty + l2Penalty);
            }
            if (fitIntercept) {
                double[] ig = error.mean().getData();
                double[] b = intercept.getData();
                b[0] -= learningRate * ig[0];
            }
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        Tensor pred = X.matmul(coefficients);
        if (fitIntercept) pred = pred.add(intercept);
        return pred;
    }

    private double sign(double x) {
        if (x > 0) return 1.0;
        if (x < 0) return -1.0;
        return 0.0;
    }

    public Tensor getCoefficients() { return coefficients; }
    public Tensor getIntercept() { return intercept; }
}
