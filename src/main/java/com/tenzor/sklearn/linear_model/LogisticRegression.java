package com.tenzor.sklearn.linear_model;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Classifier;

public class LogisticRegression extends Classifier {
    private double learningRate;
    private int maxIter;
    private double lambda;
    private Tensor weights;
    private Tensor bias;

    public LogisticRegression() { this(0.01, 1000, 0.0); }

    public LogisticRegression(double learningRate, int maxIter, double lambda) {
        super();
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        this.lambda = lambda;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        weights = Tensor.zeros(m, 1);
        bias = Tensor.zeros(1, 1);

        for (int iter = 0; iter < maxIter; iter++) {
            Tensor linear = X.matmul(weights).add(bias);
            Tensor predictions = linear.sigmoid();
            Tensor error = predictions.sub(y);
            Tensor gradW = X.transpose().matmul(error).div(n).add(weights.mul(lambda / n));
            Tensor gradB = error.sum().div(n);
            weights = weights.sub(gradW.mul(learningRate));
            bias = bias.sub(gradB.mul(learningRate));
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        Tensor proba = predictProba(X);
        double[] p = proba.getData();
        double[] result = new double[p.length];
        for (int i = 0; i < p.length; i++) result[i] = p[i] >= 0.5 ? 1.0 : 0.0;
        return new Tensor(result, new Shape(X.shape().dim(0), 1));
    }

    @Override
    public Tensor predictProba(Tensor X) {
        if (!fitted) throw new IllegalStateException("Model not fitted yet");
        return X.matmul(weights).add(bias).sigmoid();
    }

    public Tensor getWeights() { return weights; }
    public Tensor getBias() { return bias; }
}
