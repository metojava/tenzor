package com.tenzor.callbacks;

import com.tenzor.io.Checkpoint;
import com.tenzor.nn.Sequential;

public class ModelCheckpoint extends Callback {
    private Checkpoint checkpoint;
    private String monitor;
    private boolean saveBestOnly;
    private double best;

    public ModelCheckpoint(String directory) {
        this(directory, "val_loss", true);
    }

    public ModelCheckpoint(String directory, String monitor, boolean saveBestOnly) {
        this.checkpoint = new Checkpoint(directory);
        this.monitor = monitor;
        this.saveBestOnly = saveBestOnly;
        this.best = Double.MAX_VALUE;
    }

    @Override
    public void onEpochEnd(int epoch, Sequential model, double loss, double valLoss) {
        double current = monitor.equals("val_loss") ? valLoss : loss;
        if (!saveBestOnly || current < best) {
            best = current;
            try {
                checkpoint.saveLatest(model, epoch, current);
                System.out.println("Checkpoint saved to " + checkpoint);
            } catch (Exception e) {
                System.err.println("Failed to save checkpoint: " + e.getMessage());
            }
        }
    }
}
