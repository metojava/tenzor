package com.tenzor.sklearn.linear_model;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Regressor;

public class Lasso extends Regressor {
    private double alpha;
    private double learningRate;
    private int maxIter;
    private Tensor coefficients;
    private Tensor intercept;
    private boolean fitIntercept;

    public Lasso() { this(1.0, true); }

    public Lasso(double alpha, boolean fitIntercept) {
        this(alpha, fitIntercept, 0.01, 1000);
    }

    public Lasso(double alpha, boolean fitIntercept, double learningRate, int maxIter) {
        super();
        this.alpha = alpha;
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
                c[i] = c[i] - learningRate * (g[i] + alpha * sign(c[i]));
            }
            if (fitIntercept) {
                Tensor interceptGrad = error.mean();
                double[] ig = interceptGrad.getData();
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
