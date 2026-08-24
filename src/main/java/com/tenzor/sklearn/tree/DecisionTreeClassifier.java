package com.tenzor.sklearn.tree;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Classifier;
import java.util.*;

public class DecisionTreeClassifier extends Classifier {
    private int maxDepth;
    private int minSamplesSplit;
    private TreeNode root;
    private int nClasses;

    public DecisionTreeClassifier() { this(10, 2); }

    public DecisionTreeClassifier(int maxDepth, int minSamplesSplit) {
        super();
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
    }

    static class TreeNode {
        int featureIndex = -1;
        double threshold;
        TreeNode left, right;
        Double prediction;
        int depth;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        double[][] xData = X.to2D();
        double[] yData = y.getData();
        Set<Double> classes = new HashSet<>();
        for (double d : yData) classes.add(d);
        nClasses = classes.size();
        root = buildTree(xData, yData, 0);
        fitted = true;
    }

    private TreeNode buildTree(double[][] X, double[] y, int depth) {
        TreeNode node = new TreeNode();
        node.depth = depth;

        if (depth >= maxDepth || X.length < minSamplesSplit || allSame(y)) {
            node.prediction = majorityClass(y);
            return node;
        }

        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGini = Double.MAX_VALUE;

        int nFeatures = X[0].length;
        for (int f = 0; f < nFeatures; f++) {
            double[] values = new double[X.length];
            for (int i = 0; i < X.length; i++) values[i] = X[i][f];
            Arrays.sort(values);
            for (int i = 0; i < values.length - 1; i++) {
                double threshold = (values[i] + values[i + 1]) / 2;
                double gini = computeGiniSplit(X, y, f, threshold);
                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature == -1) {
            node.prediction = majorityClass(y);
            return node;
        }

        node.featureIndex = bestFeature;
        node.threshold = bestThreshold;

        List<double[]> leftX = new ArrayList<>();
        List<Double> leftY = new ArrayList<>();
        List<double[]> rightX = new ArrayList<>();
        List<Double> rightY = new ArrayList<>();

        for (int i = 0; i < X.length; i++) {
            if (X[i][bestFeature] <= bestThreshold) {
                leftX.add(X[i]);
                leftY.add(y[i]);
            } else {
                rightX.add(X[i]);
                rightY.add(y[i]);
            }
        }

        if (leftX.isEmpty() || rightX.isEmpty()) {
            node.prediction = majorityClass(y);
            return node;
        }

        node.left = buildTree(leftX.toArray(new double[0][]), leftY.stream().mapToDouble(d -> d).toArray(), depth + 1);
        node.right = buildTree(rightX.toArray(new double[0][]), rightY.stream().mapToDouble(d -> d).toArray(), depth + 1);

        return node;
    }

    private double computeGiniSplit(double[][] X, double[] y, int feature, double threshold) {
        Map<Double, Integer> leftCounts = new HashMap<>();
        Map<Double, Integer> rightCounts = new HashMap<>();
        int leftTotal = 0, rightTotal = 0;

        for (int i = 0; i < X.length; i++) {
            double label = y[i];
            if (X[i][feature] <= threshold) {
                leftCounts.merge(label, 1, Integer::sum);
                leftTotal++;
            } else {
                rightCounts.merge(label, 1, Integer::sum);
                rightTotal++;
            }
        }

        double giniLeft = 1.0;
        for (int count : leftCounts.values()) {
            double p = (double) count / leftTotal;
            giniLeft -= p * p;
        }

        double giniRight = 1.0;
        for (int count : rightCounts.values()) {
            double p = (double) count / rightTotal;
            giniRight -= p * p;
        }

        return (leftTotal * giniLeft + rightTotal * giniRight) / (leftTotal + rightTotal);
    }

    private boolean allSame(double[] y) {
        for (int i = 1; i < y.length; i++) {
            if (y[i] != y[0]) return false;
        }
        return true;
    }

    private double majorityClass(double[] y) {
        Map<Double, Integer> counts = new HashMap<>();
        for (double d : y) counts.merge(d, 1, Integer::sum);
        return counts.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    @Override
    public Tensor predict(Tensor X) {
        double[][] xData = X.to2D();
        double[] result = new double[X.shape().dim(0)];
        for (int i = 0; i < result.length; i++) {
            result[i] = predictSample(root, xData[i]);
        }
        return new Tensor(result, new Shape(result.length));
    }

    private double predictSample(TreeNode node, double[] sample) {
        if (node.prediction != null) return node.prediction;
        if (sample[node.featureIndex] <= node.threshold) {
            return predictSample(node.left, sample);
        } else {
            return predictSample(node.right, sample);
        }
    }

    @Override
    public Tensor predictProba(Tensor X) {
        return predict(X);
    }
}
