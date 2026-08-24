package com.tenzor;

import com.tenzor.core.Tensor;
import com.tenzor.sklearn.linear_model.LinearRegression;
import com.tenzor.sklearn.linear_model.LogisticRegression;
import com.tenzor.sklearn.tree.DecisionTreeClassifier;
import com.tenzor.sklearn.neighbors.KNeighborsClassifier;
import com.tenzor.sklearn.decomposition.PCA;
import com.tenzor.sklearn.preprocessing.StandardScaler;
import com.tenzor.sklearn.model_selection.train_test_split;
import com.tenzor.sklearn.model_selection.KFold;
import com.tenzor.sklearn.model_selection.cross_val_score;
import com.tenzor.metrics.R2Score;
import com.tenzor.metrics.Accuracy;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;

public class SklearnTest {

    @Test
    public void testLinearRegression() {
        double[][] xData = {{1}, {2}, {3}, {4}, {5}};
        double[] yData = {2, 4, 6, 8, 10};
        Tensor x = new Tensor(xData);
        Tensor y = new Tensor(yData, new com.tenzor.core.Shape(5, 1));

        LinearRegression lr = new LinearRegression();
        lr.fit(x, y);

        Tensor pred = lr.predict(x);
        double r2 = R2Score.compute(y, pred);
        assertTrue(r2 > 0.99, "Linear regression R² should be near 1.0");
    }

    @Test
    public void testLogisticRegression() {
        double[][] xData = {{0.1}, {0.2}, {0.8}, {0.9}};
        double[] yData = {0, 0, 1, 1};
        Tensor x = new Tensor(xData);
        Tensor y = new Tensor(yData, new com.tenzor.core.Shape(4, 1));

        LogisticRegression lr = new LogisticRegression(0.1, 500, 0.0);
        lr.fit(x, y);

        Tensor pred = lr.predict(x);
        double acc = Accuracy.compute(y, pred);
        assertTrue(acc >= 0.75, "Logistic regression accuracy");
    }

    @Test
    public void testDecisionTree() {
        double[][] xData = {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {10, 10}, {10, 11}};
        double[] yData = {0, 0, 0, 0, 1, 1};
        Tensor x = new Tensor(xData);
        Tensor y = new Tensor(yData, new com.tenzor.core.Shape(6));

        DecisionTreeClassifier dt = new DecisionTreeClassifier(5, 1);
        dt.fit(x, y);

        Tensor pred = dt.predict(x);
        double acc = Accuracy.compute(y, pred);
        assertEquals(1.0, acc, "Decision tree should overfit small data");
    }

    @Test
    public void testKNN() {
        double[][] xData = {{1, 1}, {1, 2}, {10, 10}, {10, 11}};
        double[] yData = {0, 0, 1, 1};
        Tensor x = new Tensor(xData);
        Tensor y = new Tensor(yData, new com.tenzor.core.Shape(4));

        KNeighborsClassifier knn = new KNeighborsClassifier(2);
        knn.fit(x, y);

        Tensor pred = knn.predict(x);
        double acc = Accuracy.compute(y, pred);
        assertTrue(acc > 0.5);
    }

    @Test
    public void testPCA() {
        double[][] xData = new double[20][3];
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 3; j++) {
                xData[i][j] = i + j * 0.5;
            }
        }
        Tensor x = new Tensor(xData);

        PCA pca = new PCA(2);
        pca.fit(x, null);

        Tensor transformed = pca.transform(x);
        assertEquals(2, transformed.shape().dim(1));
        assertNotNull(pca.getComponents());
        assertNotNull(pca.getExplainedVarianceRatio());
    }

    @Test
    public void testStandardScaler() {
        Tensor x = new Tensor(new double[][]{{1, 10}, {2, 20}, {3, 30}});

        StandardScaler scaler = new StandardScaler();
        Tensor transformed = scaler.fitTransform(x);

        Tensor mean = transformed.mean(0);
        assertEquals(0.0, mean.get(0), 1e-10);
        assertEquals(0.0, mean.get(1), 1e-10);

        Tensor reconstructed = scaler.inverseTransform(transformed);
        for (int i = 0; i < x.size(); i++) {
            assertEquals(x.getData()[i], reconstructed.getData()[i], 1e-10);
        }
    }

    @Test
    public void testTrainTestSplit() {
        Tensor x = Tensor.random(100, 5);
        Tensor y = Tensor.random(100, 1);

        Tensor[] split = train_test_split.split(x, y, 0.2, 42);
        assertEquals(80, split[0].shape().dim(0));
        assertEquals(20, split[3].shape().dim(0));
    }

    @Test
    public void testKFold() {
        KFold kfold = new KFold(5, true, 42);
        List<int[][]> folds = kfold.split(100);
        assertEquals(5, folds.size());
        assertEquals(80, folds.get(0)[0].length);
        assertEquals(20, folds.get(0)[1].length);
    }
}
