package com.tenzor.sklearn.model_selection;

import com.tenzor.core.Tensor;
import com.tenzor.sklearn.base.Estimator;
import java.util.*;

public class GridSearchCV {
    private Estimator estimator;
    private Map<String, double[]> paramGrid;
    private int folds;
    private KFold kfold;
    private double bestScore;
    private Map<String, Double> bestParams;
    private List<Map<String, Double>> cvResults;

    public GridSearchCV(Estimator estimator, Map<String, double[]> paramGrid, int folds) {
        this.estimator = estimator;
        this.paramGrid = paramGrid;
        this.folds = folds;
        this.kfold = new KFold(folds, true, 42);
        this.cvResults = new ArrayList<>();
    }

    public void fit(Tensor X, Tensor y) {
        List<List<Object>> paramCombinations = new ArrayList<>();
        generateCombinations(new ArrayList<>(paramGrid.keySet()), 0, new HashMap<>(), paramCombinations);

        bestScore = -Double.MAX_VALUE;

        for (List<Object> combination : paramCombinations) {
            @SuppressWarnings("unchecked")
            Map<String, Double> params = (Map<String, Double>) combination.get(0);
            List<int[][]> splits = kfold.split(X.shape().dim(0));
            double foldScore = 0;

            for (int[][] split : splits) {
                int[] trainIdx = split[0];
                int[] testIdx = split[1];
                Estimator clone = cloneEstimator(estimator);
                setParams(clone, params);

                Tensor xTrain = gatherRows(X, trainIdx);
                Tensor yTrain = gatherRows(y, trainIdx);
                Tensor xTest = gatherRows(X, testIdx);
                Tensor yTest = gatherRows(y, testIdx);

                clone.fit(xTrain, yTrain);
                foldScore += score(clone, xTest, yTest);
            }

            foldScore /= folds;
            Map<String, Double> result = new HashMap<>(params);
            result.put("mean_test_score", foldScore);
            cvResults.add(result);

            if (foldScore > bestScore) {
                bestScore = foldScore;
                bestParams = new HashMap<>(params);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void generateCombinations(List<String> keys, int depth, Map<String, Double> current,
                                       List<List<Object>> results) {
        if (depth == keys.size()) {
            Map<String, Double> copy = new HashMap<>(current);
            List<Object> wrapper = new ArrayList<>();
            wrapper.add(copy);
            results.add(wrapper);
            return;
        }

        String key = keys.get(depth);
        double[] values = paramGrid.get(key);
        for (double value : values) {
            current.put(key, value);
            generateCombinations(keys, depth + 1, current, results);
        }
    }

    private void setParams(Estimator est, Map<String, Double> params) {
        for (Map.Entry<String, Double> entry : params.entrySet()) {
            try {
                String setter = "set" + entry.getKey().substring(0, 1).toUpperCase() + entry.getKey().substring(1);
                java.lang.reflect.Method method = est.getClass().getMethod(setter, double.class);
                method.invoke(est, entry.getValue());
            } catch (Exception e) {
                try {
                    java.lang.reflect.Field field = est.getClass().getDeclaredField(entry.getKey());
                    field.setAccessible(true);
                    if (field.getType() == int.class) {
                        field.setInt(est, entry.getValue().intValue());
                    } else if (field.getType() == double.class) {
                        field.setDouble(est, entry.getValue());
                    }
                } catch (Exception ex) {
                    // ignore
                }
            }
        }
    }

    private Estimator cloneEstimator(Estimator est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator: " + est.getClass().getName(), e);
        }
    }

    private double score(Estimator est, Tensor X, Tensor y) {
        try {
            java.lang.reflect.Method scoreMethod = est.getClass().getMethod("score", Tensor.class, Tensor.class);
            return (double) scoreMethod.invoke(est, X, y);
        } catch (Exception e) {
            return 0;
        }
    }

    private Tensor gatherRows(Tensor source, int[] indices) {
        int[] dims = source.shape().dims();
        int rowSize = 1;
        for (int i = 1; i < dims.length; i++) rowSize *= dims[i];
        double[] srcData = source.getData();
        double[] resultData = new double[indices.length * rowSize];
        for (int i = 0; i < indices.length; i++) {
            System.arraycopy(srcData, indices[i] * rowSize, resultData, i * rowSize, rowSize);
        }
        int[] newDims = new int[dims.length];
        newDims[0] = indices.length;
        System.arraycopy(dims, 1, newDims, 1, dims.length - 1);
        return new Tensor(resultData, new com.tenzor.core.Shape(newDims));
    }

    public double getBestScore() { return bestScore; }
    public Map<String, Double> getBestParams() { return bestParams; }
    public List<Map<String, Double>> getCvResults() { return cvResults; }
}
