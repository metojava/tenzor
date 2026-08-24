package com.tenzor.examples;

import com.tenzor.core.Tensor;
import com.tenzor.datasets.Iris;
import com.tenzor.sklearn.ensemble.RandomForestClassifier;
import com.tenzor.metrics.Accuracy;

public class SklearnRandomForestExample {
    public static void main(String[] args) throws Exception {
        System.out.println("Loading Iris...");
        Iris dataset = new Iris();
        Tensor xTrain = dataset.getXTrain();
        Tensor yTrain = dataset.getYTrain();
        Tensor xTest = dataset.getXTest();
        Tensor yTest = dataset.getYTest();

        RandomForestClassifier rf = new RandomForestClassifier(50, 5, 2);
        rf.fit(xTrain, yTrain);

        Tensor predictions = rf.predict(xTest);
        double accuracy = Accuracy.compute(yTest, predictions);
        System.out.println("Random Forest Accuracy: " + (accuracy * 100) + "%");
    }
}
