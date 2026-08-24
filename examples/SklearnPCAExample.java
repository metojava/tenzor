package com.tenzor.examples;

import com.tenzor.core.Tensor;
import com.tenzor.datasets.Iris;
import com.tenzor.sklearn.decomposition.PCA;
import com.tenzor.sklearn.preprocessing.StandardScaler;

public class SklearnPCAExample {
    public static void main(String[] args) throws Exception {
        System.out.println("Loading Iris...");
        Iris dataset = new Iris();
        Tensor xTrain = dataset.getXTrain();

        StandardScaler scaler = new StandardScaler();
        xTrain = scaler.fitTransform(xTrain);

        PCA pca = new PCA(2);
        pca.fit(xTrain, null);

        Tensor xTransformed = pca.transform(xTrain);
        System.out.println("Original shape: " + xTrain.shape());
        System.out.println("Transformed shape: " + xTransformed.shape());
        System.out.println("Explained variance ratio: " + pca.getExplainedVarianceRatio());

        Tensor xReconstructed = pca.inverseTransform(xTransformed);
        double reconstructionError = xTrain.sub(xReconstructed).pow(2).mean().scalar();
        System.out.println("Reconstruction error: " + reconstructionError);
    }
}
