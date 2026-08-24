package com.tenzor.sklearn.linear_model;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.core.TensorMath;
import com.tenzor.sklearn.base.Regressor;

public class LinearRegression extends Regressor {
    private Tensor coefficients;
    private Tensor intercept;
    private boolean fitIntercept;

    public LinearRegression() { this(true); }

    public LinearRegression(boolean fitIntercept) {
        super();
        this.fitIntercept = fitIntercept;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);

        Tensor xData;
        if (fitIntercept) {
            Tensor ones = Tensor.ones(n, 1);
            xData = TensorMath.concatenate(ones, X, 1);
        } else {
            xData = X;
        }

        Tensor xt = xData.transpose();
        Tensor xtx = xt.matmul(xData);
        Tensor xtxInv = inverse(xtx);
        Tensor theta = xtxInv.matmul(xt).matmul(y);

        if (fitIntercept) {
            intercept = new Tensor(theta.get(0));
            double[] coefData = new double[theta.size() - 1];
            System.arraycopy(theta.getData(), 1, coefData, 0, coefData.length);
            coefficients = new Tensor(coefData, new Shape(m, 1));
        } else {
            intercept = Tensor.zeros(1);
            coefficients = theta;
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        if (!fitted) throw new IllegalStateException("Model not fitted yet");
        Tensor pred = X.matmul(coefficients);
        if (fitIntercept) {
            double[] p = pred.getData();
            double[] result = new double[p.length];
            for (int i = 0; i < p.length; i++) result[i] = p[i] + intercept.get(0);
            pred = new Tensor(result, pred.shape());
        }
        return pred;
    }

    public Tensor getCoefficients() { return coefficients; }
    public Tensor getIntercept() { return intercept; }

    private Tensor inverse(Tensor m) {
        int n = m.shape().dim(0);
        double[][] a = m.to2D();
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) inv[i][i] = 1;

        for (int col = 0; col < n; col++) {
            double maxVal = Math.abs(a[col][col]);
            int maxRow = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > maxVal) {
                    maxVal = Math.abs(a[row][col]);
                    maxRow = row;
                }
            }
            double[] temp = a[col]; a[col] = a[maxRow]; a[maxRow] = temp;
            temp = inv[col]; inv[col] = inv[maxRow]; inv[maxRow] = temp;
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
        for (int i = 0; i < n; i++)
            System.arraycopy(inv[i], 0, flat, i * n, n);
        return new Tensor(flat, new Shape(n, n));
    }
}
