package com.tenzor.sklearn.ensemble;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Classifier;
import java.util.*;

public class VotingClassifier extends Classifier {
    private List<Classifier> classifiers;
    private String voting;

    public VotingClassifier(List<Classifier> classifiers) {
        this(classifiers, "hard");
    }

    public VotingClassifier(List<Classifier> classifiers, String voting) {
        super();
        this.classifiers = classifiers;
        this.voting = voting;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        for (Classifier c : classifiers) c.fit(X, y);
        fitted = true;
    }

    @Override
    public Tensor predict(Tensor X) {
        int n = X.shape().dim(0);
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            Map<Double, Integer> votes = new HashMap<>();
            for (Classifier c : classifiers) {
                Tensor sample = X.slice(0, i, i + 1);
                double pred = c.predict(sample).get(0);
                votes.merge(pred, 1, Integer::sum);
            }
            result[i] = votes.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        }
        return new Tensor(result, new Shape(n));
    }

    @Override
    public Tensor predictProba(Tensor X) { return predict(X); }
}
