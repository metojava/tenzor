package com.tenzor.sklearn.model_selection;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.base.Estimator;
import java.util.*;

public final class cross_val_score {

    private cross_val_score() {}

    public static double[] evaluate(Estimator estimator, Tensor X, Tensor y, int folds) {
        KFold kfold = new KFold(folds, true, 42);
        List<int[][]> splits = kfold.split(X.shape().dim(0));
        double[] scores = new double[folds];

        for (int f = 0; f < splits.size(); f++) {
            int[][] split = splits.get(f);
            int[] trainIdx = split[0];
            int[] testIdx = split[1];

            Estimator clone = cloneEstimator(estimator);
            Tensor xTrain = gatherRows(X, trainIdx);
            Tensor yTrain = gatherRows(y, trainIdx);
            Tensor xTest = gatherRows(X, testIdx);
            Tensor yTest = gatherRows(y, testIdx);

            clone.fit(xTrain, yTrain);

            try {
                java.lang.reflect.Method scoreMethod = estimator.getClass().getMethod("score", Tensor.class, Tensor.class);
                scores[f] = (double) scoreMethod.invoke(clone, xTest, yTest);
            } catch (Exception e) {
                scores[f] = 0;
            }
        }

        return scores;
    }

    private static Estimator cloneEstimator(Estimator est) {
        try {
            return est.getClass().getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Cannot clone estimator", e);
        }
    }

    private static Tensor gatherRows(Tensor source, int[] indices) {
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
        return new Tensor(resultData, new Shape(newDims));
    }
}
