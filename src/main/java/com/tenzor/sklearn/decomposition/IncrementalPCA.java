package com.tenzor.sklearn.decomposition;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;

public class IncrementalPCA extends Transformer {
    private int nComponents;
    private Tensor components;
    private Tensor mean;
    private Tensor explainedVariance;
    private Tensor explainedVarianceRatio;
    private int nSamplesSeen;

    public IncrementalPCA() { this(2); }

    public IncrementalPCA(int nComponents) {
        super();
        this.nComponents = nComponents;
        this.nSamplesSeen = 0;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        int k = Math.min(nComponents, m);

        if (nSamplesSeen == 0) {
            mean = Tensor.zeros(m);
            components = Tensor.zeros(new Shape(k, m));
            explainedVariance = Tensor.zeros(k);
        }

        Tensor batchMean = X.mean(0);
        double totalN = nSamplesSeen + n;

        Tensor newMean = mean.mul(nSamplesSeen / totalN).add(batchMean.mul(n / totalN));
        Tensor centered = X.sub(batchMean);

        if (nSamplesSeen > 0) {
            Tensor oldCentered = Tensor.zeros(new Shape(nSamplesSeen, m));
            Tensor combined = com.tenzor.core.TensorMath.concatenate(oldCentered, centered, 0);
            PCA pca = new PCA(k);
            pca.fit(combined, null);
            components = pca.getComponents();
            explainedVariance = pca.getExplainedVariance();
        } else {
            PCA pca = new PCA(k);
            pca.fit(centered, null);
            components = pca.getComponents();
            explainedVariance = pca.getExplainedVariance();
        }

        mean = newMean;
        nSamplesSeen = (int) totalN;

        double totalVar = 0;
        for (int i = 0; i < m; i++) {
            totalVar += explainedVariance.size() > i ? explainedVariance.get(i) : 0;
        }
        double[] ratio = new double[explainedVariance.size()];
        for (int i = 0; i < ratio.length; i++) ratio[i] = explainedVariance.get(i) / (totalVar + 1e-10);
        explainedVarianceRatio = new Tensor(ratio, new Shape(explainedVariance.size()));

        fitted = true;
    }

    @Override
    public Tensor transform(Tensor X) {
        if (!fitted) throw new IllegalStateException("IncrementalPCA not fitted");
        return X.sub(mean).matmul(components.transpose());
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        fit(X, null);
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) {
        if (!fitted) throw new IllegalStateException("IncrementalPCA not fitted");
        return X.matmul(components).add(mean);
    }
}
