package com.tenzor.sklearn.preprocessing;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;
import java.util.*;

public class LabelEncoder extends Transformer {
    private Map<Double, Integer> mapping;
    private List<Double> classes;

    @Override
    public void fit(Tensor X, Tensor y) {
        Set<Double> unique = new TreeSet<>();
        for (int i = 0; i < X.size(); i++) unique.add(X.get(i));
        classes = new ArrayList<>(unique);
        mapping = new HashMap<>();
        for (int i = 0; i < classes.size(); i++) mapping.put(classes.get(i), i);
        fitted = true;
    }

    @Override
    public Tensor transform(Tensor X) {
        if (!fitted) throw new IllegalStateException("LabelEncoder not fitted");
        double[] result = new double[X.size()];
        for (int i = 0; i < X.size(); i++) {
            Integer idx = mapping.get(X.get(i));
            result[i] = idx != null ? idx : -1;
        }
        return new Tensor(result, X.shape());
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        fit(X, null);
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) {
        if (!fitted) throw new IllegalStateException("LabelEncoder not fitted");
        double[] result = new double[X.size()];
        for (int i = 0; i < X.size(); i++) {
            int idx = (int) Math.round(X.get(i));
            if (idx >= 0 && idx < classes.size()) result[i] = classes.get(idx);
            else result[i] = -1;
        }
        return new Tensor(result, X.shape());
    }
}
