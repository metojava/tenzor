package com.tenzor.callbacks;

import com.tenzor.nn.Sequential;
import java.util.*;

public class History extends Callback {
    private Map<String, List<Double>> metrics;

    public History() {
        this.metrics = new LinkedHashMap<>();
    }

    @Override
    public void onEpochEnd(int epoch, Sequential model, double loss, double valLoss) {
        metrics.computeIfAbsent("loss", k -> new ArrayList<>()).add(loss);
        metrics.computeIfAbsent("val_loss", k -> new ArrayList<>()).add(valLoss);
    }

    public Map<String, List<Double>> getMetrics() { return metrics; }
}
