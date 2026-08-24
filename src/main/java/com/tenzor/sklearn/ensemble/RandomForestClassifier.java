package com.tenzor.sklearn.ensemble;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Classifier;
import com.tenzor.sklearn.tree.DecisionTreeClassifier;
import java.util.*;

public class RandomForestClassifier extends Classifier {
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int maxFeatures;
    private List<DecisionTreeClassifier> trees;
    private Random rng;

    public RandomForestClassifier() { this(100, 10, 2); }

    public RandomForestClassifier(int nEstimators, int maxDepth, int minSamplesSplit) {
        super();
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.rng = new Random();
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        maxFeatures = (int) Math.sqrt(m);
        if (maxFeatures < 1) maxFeatures = 1;
        trees = new ArrayList<>();

        for (int t = 0; t < nEstimators; t++) {
            double[][] bootstrapX = new double[n][m];
            double[] bootstrapY = new double[n];
            for (int i = 0; i < n; i++) {
                int idx = rng.nextInt(n);
                bootstrapX[i] = X.to2D()[idx];
                bootstrapY[i] = y.get(idx);
            }
            DecisionTreeClassifier tree = new DecisionTreeClassifier(maxDepth, minSamplesSplit) {
                @Override
                public void fit(Tensor X, Tensor y) {
                    super.fit(X, y);
                }
            };
            Tensor xTensor = new Tensor(bootstrapX);
            Tensor yTensor = new Tensor(bootstrapY);
            tree.fit(xTensor, yTensor);
            trees.add(tree);
        }
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        int n = X.shape().dim(0);
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            Map<Double, Integer> votes = new HashMap<>();
            for (DecisionTreeClassifier tree : trees) {
                Tensor sample = X.slice(0, i, i + 1);
                double pred = tree.predict(sample).get(0);
                votes.merge(pred, 1, Integer::sum);
            }
            result[i] = votes.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        }
        return new Tensor(result, new Shape(n));
    }

    @Override
    public Tensor predictProba(Tensor X) {
        return predict(X);
    }
}
