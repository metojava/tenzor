package com.tenzor.sklearn.ensemble;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Regressor;
import com.tenzor.sklearn.tree.DecisionTreeRegressor;
import java.util.*;

public class RandomForestRegressor extends Regressor {
    private int nEstimators;
    private int maxDepth;
    private List<DecisionTreeRegressor> trees;
    private Random rng;

    public RandomForestRegressor() { this(100, 10); }

    public RandomForestRegressor(int nEstimators, int maxDepth) {
        super();
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.rng = new Random();
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        trees = new ArrayList<>();

        for (int t = 0; t < nEstimators; t++) {
            double[][] bootstrapX = new double[n][m];
            double[] bootstrapY = new double[n];
            for (int i = 0; i < n; i++) {
                int idx = rng.nextInt(n);
                bootstrapX[i] = X.to2D()[idx];
                bootstrapY[i] = y.get(idx);
            }
            DecisionTreeRegressor tree = new DecisionTreeRegressor(maxDepth, 2);
            tree.fit(new Tensor(bootstrapX), new Tensor(bootstrapY));
            trees.add(tree);
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        int n = X.shape().dim(0);
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (DecisionTreeRegressor tree : trees) {
                Tensor sample = X.slice(0, i, i + 1);
                sum += tree.predict(sample).get(0);
            }
            result[i] = sum / trees.size();
        }
        return new Tensor(result, new Shape(n));
    }
}
