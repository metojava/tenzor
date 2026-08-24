package com.tenzor.sklearn.svm;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Classifier;
import java.util.*;

public class SVC extends Classifier {
    private double C;
    private double learningRate;
    private int maxIter;
    private double tolerance;
    private Tensor weights;
    private Tensor bias;

    public SVC() { this(1.0, 0.001, 1000, 1e-4); }

    public SVC(double C, double learningRate, int maxIter, double tolerance) {
        super();
        this.C = C;
        this.learningRate = learningRate;
        this.maxIter = maxIter;
        this.tolerance = tolerance;
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
            double totalLoss = 0;
            for (int i = 0; i < n; i++) {
                double[] xi = xData[i];
                double yi = yData[i] == 0 ? -1 : 1;
                double score = 0;
                for (int j = 0; j < m; j++) score += weights.get(j) * xi[j];
                score += bias.get(0);

                double margin = yi * score;
                if (margin < 1) {
                    double[] w = weights.getData();
                    double[] b = bias.getData();
                    for (int j = 0; j < m; j++) {
                        w[j] -= learningRate * (w[j] - C * yi * xi[j]);
                    }
                    b[0] -= learningRate * (-C * yi);
                    totalLoss += (1 - margin);
                } else {
                    double[] w = weights.getData();
                    for (int j = 0; j < m; j++) {
                        w[j] -= learningRate * w[j];
                    }
                }
            }
            if (totalLoss / n < tolerance) break;
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        double[] result = new double[X.shape().dim(0)];
        double[][] xData = X.to2D();
        for (int i = 0; i < result.length; i++) {
            double score = 0;
            for (int j = 0; j < weights.size(); j++) score += weights.get(j) * xData[i][j];
            score += bias.get(0);
            result[i] = score >= 0 ? 1.0 : 0.0;
        }
        return new Tensor(result, new Shape(result.length));
    }

    @Override
    public Tensor predictProba(Tensor X) { return predict(X); }
}
