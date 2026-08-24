package com.tenzor.sklearn.ensemble;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Regressor;
import com.tenzor.sklearn.tree.DecisionTreeRegressor;
import java.util.*;

public class GradientBoostingRegressor extends Regressor {
    private int nEstimators;
    private double learningRate;
    private int maxDepth;
    private List<DecisionTreeRegressor> trees;
    private double initialPrediction;

    public GradientBoostingRegressor() { this(100, 0.1, 3); }

    public GradientBoostingRegressor(int nEstimators, double learningRate, int maxDepth) {
        super();
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        double[] yData = y.getData();
        double sum = 0;
        for (double v : yData) sum += v;
        initialPrediction = sum / n;
        trees = new ArrayList<>();

        double[] F = new double[n];
        Arrays.fill(F, initialPrediction);

        for (int t = 0; t < nEstimators; t++) {
            double[] residuals = new double[n];
            for (int i = 0; i < n; i++) residuals[i] = yData[i] - F[i];

            DecisionTreeRegressor tree = new DecisionTreeRegressor(maxDepth, 2);
            tree.fit(X, new Tensor(residuals, new Shape(n)));
            trees.add(tree);

            double[] preds = tree.predict(X).getData();
            for (int i = 0; i < n; i++) F[i] += learningRate * preds[i];
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        int n = X.shape().dim(0);
        double[] F = new double[n];
        Arrays.fill(F, initialPrediction);
        for (DecisionTreeRegressor tree : trees) {
            double[] preds = tree.predict(X).getData();
            for (int i = 0; i < n; i++) F[i] += learningRate * preds[i];
        }
        return new Tensor(F, new Shape(n));
    }
}
