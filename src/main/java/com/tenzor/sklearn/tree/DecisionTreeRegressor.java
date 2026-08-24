package com.tenzor.sklearn.tree;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Regressor;
import java.util.*;

public class DecisionTreeRegressor extends Regressor {
    private int maxDepth;
    private int minSamplesSplit;
    private TreeNode root;

    public DecisionTreeRegressor() { this(10, 2); }

    public DecisionTreeRegressor(int maxDepth, int minSamplesSplit) {
        super();
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
    }

    static class TreeNode {
        int featureIndex = -1;
        double threshold;
        TreeNode left, right;
        Double prediction;
    }

    @Override
    public void fit(Tensor X, Tensor y) {
        root = buildTree(X.to2D(), y.getData(), 0);
        fitted = true;
    }

    private TreeNode buildTree(double[][] X, double[] y, int depth) {
        TreeNode node = new TreeNode();

        if (depth >= maxDepth || X.length < minSamplesSplit) {
            double sum = 0;
            for (double v : y) sum += v;
            node.prediction = sum / y.length;
            return node;
        }

        int bestFeature = -1;
        double bestThreshold = 0;
        double bestMSE = Double.MAX_VALUE;

        int nFeatures = X[0].length;
        for (int f = 0; f < nFeatures; f++) {
            double[] values = new double[X.length];
            for (int i = 0; i < X.length; i++) values[i] = X[i][f];
            Arrays.sort(values);
            for (int i = 0; i < values.length - 1; i++) {
                double threshold = (values[i] + values[i + 1]) / 2;
                double mse = computeMSESplit(X, y, f, threshold);
                if (mse < bestMSE) {
                    bestMSE = mse;
                    bestFeature = f;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature == -1) {
            double sum = 0;
            for (double v : y) sum += v;
            node.prediction = sum / y.length;
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
                leftX.add(X[i]); leftY.add(y[i]);
            } else {
                rightX.add(X[i]); rightY.add(y[i]);
            }
        }

        if (leftX.isEmpty() || rightX.isEmpty()) {
            double sum = 0;
            for (double v : y) sum += v;
            node.prediction = sum / y.length;
            return node;
        }

        node.left = buildTree(leftX.toArray(new double[0][]), leftY.stream().mapToDouble(d -> d).toArray(), depth + 1);
        node.right = buildTree(rightX.toArray(new double[0][]), rightY.stream().mapToDouble(d -> d).toArray(), depth + 1);

        return node;
    }

    private double computeMSESplit(double[][] X, double[] y, int feature, double threshold) {
        double sumLeft = 0, sumRight = 0;
        int nLeft = 0, nRight = 0;

        for (int i = 0; i < X.length; i++) {
            if (X[i][feature] <= threshold) {
                sumLeft += y[i];
                nLeft++;
            } else {
                sumRight += y[i];
                nRight++;
            }
        }

        double meanLeft = nLeft > 0 ? sumLeft / nLeft : 0;
        double meanRight = nRight > 0 ? sumRight / nRight : 0;

        double mseLeft = 0, mseRight = 0;
        for (int i = 0; i < X.length; i++) {
            if (X[i][feature] <= threshold) {
                mseLeft += Math.pow(y[i] - meanLeft, 2);
            } else {
                mseRight += Math.pow(y[i] - meanRight, 2);
            }
        }

        return (mseLeft + mseRight) / X.length;
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
}
