package com.tenzor.callbacks;

import com.tenzor.nn.Sequential;

public class ReduceLROnPlateau extends Callback {
    private String monitor;
    private double factor;
    private int patience;
    private double minLr;
    private double minDelta;
    private int wait;
    private double best;
    private double currentLr;

    public ReduceLROnPlateau() {
        this("val_loss", 0.5, 5, 1e-6, 0.0001);
    }

    public ReduceLROnPlateau(String monitor, double factor, int patience, double minLr, double minDelta) {
        this.monitor = monitor;
        this.factor = factor;
        this.patience = patience;
        this.minLr = minLr;
        this.minDelta = minDelta;
        this.best = Double.MAX_VALUE;
        this.wait = 0;
    }

    @Override
    public void onEpochEnd(int epoch, Sequential model, double loss, double valLoss) {
        double current = monitor.equals("val_loss") ? valLoss : loss;
        if (current < best - minDelta) {
            best = current;
            wait = 0;
        } else {
            wait++;
            if (wait >= patience) {
                if (model.getOptimizer() != null) {
                    double lr = model.getOptimizer().getLearningRate();
                    double newLr = Math.max(lr * factor, minLr);
                    model.getOptimizer().setLearningRate(newLr);
                    System.out.println("Reduced LR to " + newLr);
                }
                wait = 0;
            }
        }
    }
}
