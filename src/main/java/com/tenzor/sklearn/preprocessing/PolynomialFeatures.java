package com.tenzor.sklearn.preprocessing;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;

public class PolynomialFeatures extends Transformer {
    private int degree;
    private boolean includeBias;

    public PolynomialFeatures() { this(2, true); }

    public PolynomialFeatures(int degree, boolean includeBias) {
        super();
        this.degree = degree;
        this.includeBias = includeBias;
    }

    @Override
    public void fit(Tensor X, Tensor y) { fitted = true; }

    @Override
    public Tensor transform(Tensor X) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        double[][] xData = X.to2D();

        int outputSize = includeBias ? 1 : 0;
        for (int d = 1; d <= degree; d++) {
            outputSize += combinations(m + d - 1, d);
        }

        double[][] result = new double[n][outputSize];
        for (int i = 0; i < n; i++) {
            int col = 0;
            if (includeBias) result[i][col++] = 1.0;
            for (int d = 1; d <= degree; d++) {
                int[] indices = new int[d];
                fillCombinations(result[i], xData[i], indices, 0, m, 0, d, col);
                col += combinations(m + d - 1, d);
            }
        }
        return new Tensor(result);
    }

    private int fillCombinations(double[] row, double[] x, int[] indices, int start, int m,
                                  int depth, int maxDepth, int col) {
        if (depth == maxDepth) {
            double val = 1.0;
            for (int i = 0; i < maxDepth; i++) val *= x[indices[i]];
            row[col] = val;
            return col + 1;
        }
        for (int i = start; i < m; i++) {
            indices[depth] = i;
            col = fillCombinations(row, x, indices, i, m, depth + 1, maxDepth, col);
        }
        return col;
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) { return X; }

    private int combinations(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n - k) k = n - k;
        int result = 1;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }
}
