package com.tenzor.callbacks;

import com.tenzor.nn.Sequential;

public abstract class Callback {
    protected boolean stopTraining = false;

    public void onTrainBegin(Sequential model) {}
    public void onTrainEnd(Sequential model) {}
    public void onEpochBegin(int epoch, Sequential model) {}
    public void onEpochEnd(int epoch, Sequential model, double loss, double valLoss) {}
    public void onBatchBegin(int batch, Sequential model) {}
    public void onBatchEnd(int batch, Sequential model, double loss) {}

    public boolean shouldStop() { return stopTraining; }
}
