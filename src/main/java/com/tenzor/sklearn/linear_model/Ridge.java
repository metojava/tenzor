package com.tenzor.sklearn.linear_model;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Regressor;

public class Ridge extends Regressor {
    private double alpha;
    private boolean fitIntercept;
    private Tensor coefficients;
    private Tensor intercept;

    public Ridge() { this(1.0, true); }

    public Ridge(double alpha, boolean fitIntercept) {
        super();
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        Tensor Xt;
        if (fitIntercept) {
            Tensor ones = Tensor.ones(n, 1);
            Xt = com.tenzor.core.TensorMath.concatenate(ones, X, 1);
        } else {
            Xt = X;
        }
        int p = Xt.shape().dim(1);
        Tensor I = Tensor.eye(p);
        Tensor A = Xt.transpose().matmul(Xt).add(I.mul(alpha));
        Tensor theta = inverse(A).matmul(Xt.transpose()).matmul(y);

        if (fitIntercept) {
            intercept = new Tensor(theta.get(0));
            double[] coef = new double[theta.size() - 1];
            System.arraycopy(theta.getData(), 1, coef, 0, coef.length);
            coefficients = new Tensor(coef, new Shape(m));
        } else {
            intercept = Tensor.zeros(1);
            coefficients = theta;
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        Tensor pred = X.matmul(coefficients);
        if (fitIntercept) pred = pred.add(intercept);
        return pred;
    }

    private Tensor inverse(Tensor m) {
        int n = m.shape().dim(0);
        double[][] a = m.to2D();
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) { inv[i][i] = 1; }
        for (int col = 0; col < n; col++) {
            double div = a[col][col];
            for (int j = 0; j < n; j++) { a[col][j] /= div; inv[col][j] /= div; }
            for (int row = 0; row < n; row++) {
                if (row != col) {
                    double factor = a[row][col];
                    for (int j = 0; j < n; j++) {
                        a[row][j] -= factor * a[col][j];
                        inv[row][j] -= factor * inv[col][j];
                    }
                }
            }
        }
        double[] flat = new double[n * n];
        for (int i = 0; i < n; i++) System.arraycopy(inv[i], 0, flat, i * n, n);
        return new Tensor(flat, new Shape(n, n));
    }

    public Tensor getCoefficients() { return coefficients; }
    public Tensor getIntercept() { return intercept; }
    public void setAlpha(double alpha) { this.alpha = alpha; }
}
