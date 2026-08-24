package com.tenzor.examples;

import com.tenzor.core.Tensor;
import com.tenzor.datasets.BostonHousing;
import com.tenzor.sklearn.linear_model.LinearRegression;
import com.tenzor.sklearn.preprocessing.StandardScaler;
import com.tenzor.metrics.R2Score;
import com.tenzor.metrics.MeanAbsoluteError;

public class SklearnLinearRegressionExample {
    public static void main(String[] args) throws Exception {
        System.out.println("Loading Boston Housing...");
        BostonHousing dataset = new BostonHousing();
        Tensor xTrain = dataset.getXTrain();
        Tensor yTrain = dataset.getYTrain();
        Tensor xTest = dataset.getXTest();
        Tensor yTest = dataset.getYTest();

        StandardScaler scaler = new StandardScaler();
        xTrain = scaler.fitTransform(xTrain);
        xTest = scaler.transform(xTest);

        LinearRegression model = new LinearRegression();
        model.fit(xTrain, yTrain);

        Tensor predictions = model.predict(xTest);
        double r2 = R2Score.compute(yTest, predictions);
        double mae = MeanAbsoluteError.compute(yTest, predictions);
        System.out.println("Linear Regression Results:");
        System.out.println("  R^2 Score: " + r2);
        System.out.println("  MAE: " + mae);
    }
}
