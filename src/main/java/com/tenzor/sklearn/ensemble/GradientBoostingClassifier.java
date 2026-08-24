package com.tenzor.sklearn.ensemble;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Classifier;
import com.tenzor.sklearn.tree.DecisionTreeRegressor;
import java.util.*;

public class GradientBoostingClassifier extends Classifier {
    private int nEstimators;
    private double learningRate;
    private int maxDepth;
    private List<DecisionTreeRegressor> trees;
    private double initialPrediction;

    public GradientBoostingClassifier() { this(100, 0.1, 3); }

    public GradientBoostingClassifier(int nEstimators, double learningRate, int maxDepth) {
        super();
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        double[] yData = y.getData();
        double pos = 0;
        for (double v : yData) pos += v;
        initialPrediction = Math.log((pos + 1) / (n - pos + 1));
        trees = new ArrayList<>();

        double[] F = new double[n];
        Arrays.fill(F, initialPrediction);

        for (int t = 0; t < nEstimators; t++) {
            double[] residuals = new double[n];
            for (int i = 0; i < n; i++) {
                double p = 1.0 / (1.0 + Math.exp(-F[i]));
                residuals[i] = yData[i] - p;
            }
            DecisionTreeRegressor tree = new DecisionTreeRegressor(maxDepth, 2);
            tree.fit(X, new Tensor(residuals, new Shape(n)));
            trees.add(tree);

            double[] preds = tree.predict(X).getData();
            for (int i = 0; i < n; i++) {
                F[i] += learningRate * preds[i];
            }
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        int n = X.shape().dim(0);
        double[] result = new double[n];
        double[] F = new double[n];
        Arrays.fill(F, initialPrediction);
        for (DecisionTreeRegressor tree : trees) {
            double[] preds = tree.predict(X).getData();
            for (int i = 0; i < n; i++) F[i] += learningRate * preds[i];
        }
        for (int i = 0; i < n; i++) {
            result[i] = F[i] >= 0 ? 1.0 : 0.0;
        }
        return new Tensor(result, new Shape(n));
    }

    @Override
    public Tensor predictProba(Tensor X) {
        double[] F = new double[X.shape().dim(0)];
        Arrays.fill(F, initialPrediction);
        for (DecisionTreeRegressor tree : trees) {
            double[] preds = tree.predict(X).getData();
            for (int i = 0; i < F.length; i++) F[i] += learningRate * preds[i];
        }
        double[] result = new double[F.length];
        for (int i = 0; i < F.length; i++) result[i] = 1.0 / (1.0 + Math.exp(-F[i]));
        return new Tensor(result, new Shape(F.length));
    }
}
