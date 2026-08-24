package com.tenzor.callbacks;

import com.tenzor.nn.Sequential;

public class EarlyStopping extends Callback {
    private String monitor;
    private double minDelta;
    private int patience;
    private int bestEpoch;
    private double bestLoss;
    private int wait;

    public EarlyStopping() {
        this("val_loss", 0.001, 10);
    }

    public EarlyStopping(String monitor, double minDelta, int patience) {
        this.monitor = monitor;
        this.minDelta = minDelta;
        this.patience = patience;
        this.bestLoss = Double.MAX_VALUE;
        this.bestEpoch = 0;
        this.wait = 0;
    }

    @Override
    public void onEpochEnd(int epoch, Sequential model, double loss, double valLoss) {
        double current = monitor.equals("val_loss") ? valLoss : loss;
        if (current < bestLoss - minDelta) {
            bestLoss = current;
            bestEpoch = epoch;
            wait = 0;
        } else {
            wait++;
            if (wait >= patience) {
                System.out.println("Early stopping triggered at epoch " + (epoch + 1));
                stopTraining = true;
            }
        }
    }
}
