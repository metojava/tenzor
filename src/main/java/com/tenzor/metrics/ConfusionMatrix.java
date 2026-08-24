package com.tenzor.metrics;

import com.tenzor.core.Tensor;
import java.util.*;

public class ConfusionMatrix {
    private int[][] matrix;
    private List<Double> classes;

    public ConfusionMatrix(Tensor yTrue, Tensor yPred) {
        Set<Double> unique = new TreeSet<>();
        for (int i = 0; i < yTrue.size(); i++) {
            unique.add((double) Math.round(yTrue.get(i)));
            unique.add((double) Math.round(yPred.get(i)));
        }
        classes = new ArrayList<>(unique);
        int n = classes.size();
        matrix = new int[n][n];

        Map<Double, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) map.put(classes.get(i), i);

        for (int i = 0; i < yTrue.size(); i++) {
            int trueIdx = map.get(Math.round(yTrue.get(i)));
            int predIdx = map.get(Math.round(yPred.get(i)));
            matrix[trueIdx][predIdx]++;
        }
    }

    public int[][] getMatrix() { return matrix; }

    public List<Double> getClasses() { return classes; }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Confusion Matrix:\n    ");
        for (double c : classes) sb.append(String.format("%6.0f", c));
        sb.append("\n");
        for (int i = 0; i < matrix.length; i++) {
            sb.append(String.format("%4.0f ", classes.get(i)));
            for (int j = 0; j < matrix[i].length; j++) {
                sb.append(String.format("%6d", matrix[i][j]));
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
