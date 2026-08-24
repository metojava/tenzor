package com.tenzor.sklearn.preprocessing;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Transformer;
import java.util.*;

public class OneHotEncoder extends Transformer {
    private List<Map<Double, Integer>> categoryMaps;
    private List<Integer> categorySizes;

    public OneHotEncoder() { super(); }

    @Override
    public void fit(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        int cols = X.shape().numDims() == 1 ? 1 : X.shape().dim(1);
        categoryMaps = new ArrayList<>();
        categorySizes = new ArrayList<>();

        for (int j = 0; j < cols; j++) {
            Set<Double> cats = new TreeSet<>();
            if (cols == 1) {
                for (int i = 0; i < n; i++) cats.add(X.get(i));
            } else {
                for (int i = 0; i < n; i++) cats.add(X.get(i, j));
            }
            Map<Double, Integer> map = new HashMap<>();
            int idx = 0;
            for (double cat : cats) map.put(cat, idx++);
            categoryMaps.add(map);
            categorySizes.add(map.size());
        }
        fitted = true;
    }

    @Override
    public Tensor transform(Tensor X) {
        if (!fitted) throw new IllegalStateException("OneHotEncoder not fitted");
        int n = X.shape().dim(0);
        int cols = categoryMaps.size();
        int totalDim = 0;
        for (int s : categorySizes) totalDim += s;

        double[] result = new double[n * totalDim];
        for (int i = 0; i < n; i++) {
            int offset = 0;
            for (int j = 0; j < cols; j++) {
                double val = cols == 1 ? X.get(i) : X.get(i, j);
                Integer catIdx = categoryMaps.get(j).get(val);
                if (catIdx != null) {
                    result[i * totalDim + offset + catIdx] = 1.0;
                }
                offset += categorySizes.get(j);
            }
        }
        return new Tensor(result, new Shape(n, totalDim));
    }

    @Override
    public Tensor fitTransform(Tensor X) {
        fit(X, null);
        return transform(X);
    }

    @Override
    public Tensor inverseTransform(Tensor X) {
        if (!fitted) throw new IllegalStateException("OneHotEncoder not fitted");
        int n = X.shape().dim(0);
        int totalDim = 0;
        for (int s : categorySizes) totalDim += s;
        double[] result = new double[n];

        int offset = 0;
        for (int j = 0; j < categoryMaps.size(); j++) {
            int size = categorySizes.get(j);
            Map<Integer, Double> rev = new HashMap<>();
            for (Map.Entry<Double, Integer> e : categoryMaps.get(j).entrySet()) {
                rev.put(e.getValue(), e.getKey());
            }
            for (int i = 0; i < n; i++) {
                int bestIdx = 0;
                double bestVal = -1;
                for (int k = 0; k < size; k++) {
                    double v = X.get(i * totalDim + offset + k);
                    if (v > bestVal) { bestVal = v; bestIdx = k; }
                }
                result[i] = rev.get(bestIdx);
            }
            offset += size;
        }
        return new Tensor(result, new Shape(n));
    }
}
