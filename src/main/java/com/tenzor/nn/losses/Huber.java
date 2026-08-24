package com.tenzor.nn.losses;

import com.tenzor.core.Tensor;

public class Huber extends Loss {
    private double delta;

    public Huber() { this(1.0); }

    public Huber(double delta) {
        super("huber");
        this.delta = delta;
    }

    @Override
    public Tensor forward(Tensor predictions, Tensor targets) {
        Tensor diff = predictions.sub(targets).abs();
        double[] d = diff.getData();
        double total = 0;
        for (double v : d) {
            if (v <= delta) {
                total += 0.5 * v * v;
            } else {
                total += delta * (v - 0.5 * delta);
            }
        }
        return new Tensor(total / d.length);
    }

    @Override
    public Tensor backward(Tensor predictions, Tensor targets) {
        Tensor diff = predictions.sub(targets);
        int n = predictions.size();
        return diff.elementWiseScalar(d -> {
            double absD = Math.abs(d);
            if (absD <= delta) return d / n;
            return (d > 0 ? delta : -delta) / n;
        });
    }
}
