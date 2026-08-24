package com.tenzor;

import com.tenzor.core.Tensor;
import com.tenzor.nn.Sequential;
import com.tenzor.nn.layers.*;
import com.tenzor.nn.losses.MeanSquaredError;
import com.tenzor.nn.optimizers.SGD;
import com.tenzor.metrics.Accuracy;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ModelTrainingTest {

    @Test
    public void testXOR() {
        Tensor x = new Tensor(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Tensor y = new Tensor(new double[][]{{0}, {1}, {1}, {0}});

        Sequential model = new Sequential();
        model.add(new Dense(8, "relu"));
        model.add(new Dense(1, "sigmoid"));
        model.compile(new MeanSquaredError(), new SGD(0.5));
        model.fit(x, y, 500, 4, 0.5);

        Tensor predictions = model.predict(x);
        double accuracy = Accuracy.compute(y, predictions);
        assertTrue(accuracy >= 0.75, "XOR accuracy should be at least 75%");
    }

    @Test
    public void testModelPredictShape() {
        Tensor x = Tensor.random(10, 5);
        Tensor y = Tensor.random(10, 1);

        Sequential model = new Sequential();
        model.add(new Dense(3, "relu"));
        model.add(new Dense(1, "linear"));
        model.compile(new MeanSquaredError(), new SGD(0.01));
        model.fit(x, y, 5, 10, 0.01);

        Tensor pred = model.predict(x);
        assertEquals(10, pred.shape().dim(0));
        assertEquals(1, pred.shape().dim(1));
    }

    @Test
    public void testLayerCompatibility() {
        Sequential model = new Sequential();
        model.add(new Dense(10, "relu"));
        model.add(new Activation("relu"));
        assertEquals(2, model.getLayers().size());
    }
}
