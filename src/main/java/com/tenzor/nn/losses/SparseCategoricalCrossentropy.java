package com.tenzor.nn.losses;

import com.tenzor.core.Tensor;

public class SparseCategoricalCrossentropy extends Loss {
    private double epsilon;

    public SparseCategoricalCrossentropy() { this(1e-7); }

    public SparseCategoricalCrossentropy(double epsilon) {
        super("sparse_categorical_crossentropy");
        this.epsilon = epsilon;
    }

    @Override
    public Tensor forward(Tensor predictions, Tensor targets) {
        Tensor clipped = predictions.elementWiseScalar(
            d -> Math.max(epsilon, Math.min(1 - epsilon, d)));
        int n = predictions.shape().dim(0);
        int c = predictions.shape().dim(predictions.shape().numDims() - 1);
        double total = 0;
        double[] predData = clipped.getData();
        double[] targetData = targets.getData();
        for (int i = 0; i < n; i++) {
            int label = (int) Math.round(targetData[i]);
            if (label >= c) label = c - 1;
            if (label < 0) label = 0;
            total += -Math.log(Math.max(epsilon, predData[i * c + label]));
        }
        return new Tensor(total / n);
    }

    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        double[] predData = predictions.getData();
        double[] targetData = targets.getData();
        double[] grad = new double[predData.length];
        int n = predictions.shape().dim(0);
        int c = predictions.shape().dim(predictions.shape().numDims() - 1);
        for (int i = 0; i < n; i++) {
            int label = (int) Math.round(targetData[i]);
            if (label >= c) label = c - 1;
            if (label < 0) label = 0;
            for (int j = 0; j < c; j++) {
                double target = (j == label) ? 1.0 : 0.0;
                grad[i * c + j] = (predData[i * c + j] - target) / n;
            }
        }
        return new Tensor(grad, predictions.shape());
    }
}
