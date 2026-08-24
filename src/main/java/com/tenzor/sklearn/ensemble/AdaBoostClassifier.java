package com.tenzor.sklearn.ensemble;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Classifier;
import com.tenzor.sklearn.tree.DecisionTreeClassifier;
import java.util.*;

public class AdaBoostClassifier extends Classifier {
    private int nEstimators;
    private List<DecisionTreeClassifier> trees;
    private List<Double> alphas;
    private Random rng;

    public AdaBoostClassifier() { this(50); }

    public AdaBoostClassifier(int nEstimators) {
        super();
        this.nEstimators = nEstimators;
        this.rng = new Random();
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int m = X.shape().dim(1);
        trees = new ArrayList<>();
        alphas = new ArrayList<>();

        double[] weights = new double[n];
        Arrays.fill(weights, 1.0 / n);
        double[] yData = y.getData();

        for (int t = 0; t < nEstimators; t++) {
            double[][] sampleX = new double[n][m];
            double[] sampleY = new double[n];
            for (int i = 0; i < n; i++) {
                int idx = sampleFromWeights(weights);
                sampleX[i] = X.to2D()[idx];
                sampleY[i] = yData[idx];
            }

            DecisionTreeClassifier tree = new DecisionTreeClassifier(1, 2);
            tree.fit(new Tensor(sampleX), new Tensor(sampleY));
            trees.add(tree);

            double[] preds = tree.predict(X).getData();
            double error = 0;
            for (int i = 0; i < n; i++) {
                if (Math.round(preds[i]) != Math.round(yData[i])) {
                    error += weights[i];
                }
            }

            if (error > 0.5) { trees.remove(trees.size() - 1); break; }

            double alpha = 0.5 * Math.log((1 - error) / Math.max(error, 1e-10));
            alphas.add(alpha);

            double sum = 0;
            for (int i = 0; i < n; i++) {
                if (Math.round(preds[i]) != Math.round(yData[i])) {
                    weights[i] *= Math.exp(alpha);
                } else {
                    weights[i] *= Math.exp(-alpha);
                }
                sum += weights[i];
            }
            for (int i = 0; i < n; i++) weights[i] /= sum;
        }
        fitted = true;
    }

    private int sampleFromWeights(double[] weights) {
        double r = rng.nextDouble();
        double cum = 0;
        for (int i = 0; i < weights.length; i++) {
            cum += weights[i];
            if (r <= cum) return i;
        }
        return weights.length - 1;
    }

    @Override
    public Tensor predict(Tensor X) {
        int n = X.shape().dim(0);
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int t = 0; t < trees.size(); t++) {
                Tensor sample = X.slice(0, i, i + 1);
                double pred = trees.get(t).predict(sample).get(0);
                sum += alphas.get(t) * (pred > 0 ? 1 : -1);
            }
            result[i] = sum >= 0 ? 1.0 : 0.0;
        }
        return new Tensor(result, new Shape(n));
    }

    @Override
    public Tensor predictProba(Tensor X) { return predict(X); }
}
