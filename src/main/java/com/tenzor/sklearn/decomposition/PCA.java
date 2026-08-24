package com.tenzor.sklearn.decomposition;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;

public class PCA extends Transformer {
    private int nComponents;
    private Tensor components;
    private Tensor mean;
    private Tensor explainedVariance;
    private Tensor explainedVarianceRatio;

    public PCA() { this(2); }

    public PCA(int nComponents) {
        super();
        this.nComponents = nComponents;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);

        mean = X.mean(0);
        Tensor centered = X.sub(mean);

        Tensor cov = centered.transpose().matmul(centered).div(n - 1);

        double[][] covData = cov.to2D();
        double[][] eigVec = new double[m][m];
        double[] eigVal = new double[m];

        for (int i = 0; i < m; i++) eigVec[i][i] = 1.0;

        for (int iter = 0; iter < 100; iter++) {
            for (int i = 0; i < m; i++) {
                for (int j = i + 1; j < m; j++) {
                    double theta = (covData[j][j] - covData[i][i]) / (2 * covData[i][j] + 1e-10);
                    double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
                    double c = 1 / Math.sqrt(1 + t * t);
                    double s = c * t;

                    for (int k = 0; k < m; k++) {
                        double oldKi = covData[k][i];
                        double oldKj = covData[k][j];
                        covData[k][i] = c * oldKi - s * oldKj;
                        covData[k][j] = s * oldKi + c * oldKj;
                    }
                    for (int k = 0; k < m; k++) {
                        double oldIk = covData[i][k];
                        double oldJk = covData[j][k];
                        covData[i][k] = c * oldIk - s * oldJk;
                        covData[j][k] = s * oldIk + c * oldJk;
                    }
                    for (int k = 0; k < m; k++) {
                        double oldKi = eigVec[k][i];
                        double oldKj = eigVec[k][j];
                        eigVec[k][i] = c * oldKi - s * oldKj;
                        eigVec[k][j] = s * oldKi + c * oldKj;
                    }
                }
            }
        }

        for (int i = 0; i < m; i++) eigVal[i] = Math.abs(covData[i][i]);

        Integer[] idx = new Integer[m];
        for (int i = 0; i < m; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(eigVal[b], eigVal[a]));

        int k = Math.min(nComponents, m);
        double[] compData = new double[k * m];
        double[] eVals = new double[k];
        double totalVar = 0;
        for (int i = 0; i < m; i++) totalVar += eigVal[i];

        for (int i = 0; i < k; i++) {
            eVals[i] = eigVal[idx[i]];
            for (int j = 0; j < m; j++) {
                compData[i * m + j] = eigVec[j][idx[i]];
            }
        }

        components = new Tensor(compData, new Shape(k, m));
        explainedVariance = new Tensor(eVals, new Shape(k));

        double[] ratio = new double[k];
        for (int i = 0; i < k; i++) ratio[i] = eVals[i] / totalVar;
        explainedVarianceRatio = new Tensor(ratio, new Shape(k));

        fitted = true;
    }

    @Override
    public Tensor transform(Tensor X) {
        if (!fitted) throw new IllegalStateException("PCA not fitted");
        return X.sub(mean).matmul(components.transpose());
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        fit(X, null);
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) {
        if (!fitted) throw new IllegalStateException("PCA not fitted");
        return X.matmul(components).add(mean);
    }

    public Tensor getComponents() { return components; }
    public Tensor getMean() { return mean; }
    public Tensor getExplainedVariance() { return explainedVariance; }
    public Tensor getExplainedVarianceRatio() { return explainedVarianceRatio; }
}
