package com.tenzor.examples;

import com.tenzor.core.Tensor;
import com.tenzor.datasets.Iris;
import com.tenzor.sklearn.svm.SVC;
import com.tenzor.sklearn.preprocessing.StandardScaler;
import com.tenzor.metrics.Accuracy;
import com.tenzor.metrics.ConfusionMatrix;

public class SklearnSVMExample {
    public static void main(String[] args) throws Exception {
        System.out.println("Loading Iris...");
        Iris dataset = new Iris();
        Tensor xTrain = dataset.getXTrain();
        Tensor yTrain = dataset.getYTrain();
        Tensor xTest = dataset.getXTest();
        Tensor yTest = dataset.getYTest();

        StandardScaler scaler = new StandardScaler();
        xTrain = scaler.fitTransform(xTrain);
        xTest = scaler.transform(xTest);

        SVC svm = new SVC(1.0, 0.001, 1000, 1e-4);
        svm.fit(xTrain, yTrain);

        Tensor predictions = svm.predict(xTest);
        double accuracy = Accuracy.compute(yTest, predictions);
        System.out.println("SVM Accuracy: " + (accuracy * 100) + "%");

        ConfusionMatrix cm = new ConfusionMatrix(yTest, predictions);
        System.out.println(cm);
    }
}
