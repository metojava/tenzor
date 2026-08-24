package com.tenzor.training;

import com.tenzor.core.Tensor;
import java.util.*;
import java.io.*;

public class ExecutionMatrix implements Serializable {
    private List<EpochRecord> epochs;
    private long startTime;
    private long endTime;
    private String modelName;
    private Map<String, Object> metadata;

    public ExecutionMatrix() {
        this.epochs = new ArrayList<>();
        this.metadata = new LinkedHashMap<>();
    }

    public void startTraining() {
        this.startTime = System.currentTimeMillis();
    }

    public void endTraining() {
        this.endTime = System.currentTimeMillis();
    }

    public void setModelName(String name) {
        this.modelName = name;
    }

    public void addMetadata(String key, Object value) {
        metadata.put(key, value);
    }

    public EpochRecord startEpoch(int epochNumber) {
        EpochRecord record = new EpochRecord(epochNumber);
        record.startTime = System.currentTimeMillis();
        return record;
    }

    public void endEpoch(EpochRecord record) {
        record.endTime = System.currentTimeMillis();
        epochs.add(record);
    }

    public void addEpoch(int epochNumber, double trainLoss, double valLoss,
                          double trainAccuracy, double valAccuracy, long durationMs) {
        EpochRecord record = new EpochRecord(epochNumber);
        record.trainLoss = trainLoss;
        record.validationLoss = valLoss;
        record.trainAccuracy = trainAccuracy;
        record.validationAccuracy = valAccuracy;
        record.durationMs = durationMs;
        epochs.add(record);
    }

    public void addEpoch(int epochNumber, double trainLoss, double valLoss, long durationMs) {
        addEpoch(epochNumber, trainLoss, valLoss, Double.NaN, Double.NaN, durationMs);
    }

    public List<EpochRecord> getEpochs() {
        return epochs;
    }

    public int getNumEpochs() {
        return epochs.size();
    }

    public long getTotalTrainingTimeMs() {
        if (endTime > 0 && startTime > 0) {
            return endTime - startTime;
        }
        long total = 0;
        for (EpochRecord e : epochs) total += e.durationMs;
        return total;
    }

    public double getBestTrainLoss() {
        double best = Double.POSITIVE_INFINITY;
        for (EpochRecord e : epochs) {
            if (!Double.isNaN(e.trainLoss) && e.trainLoss < best) {
                best = e.trainLoss;
            }
        }
        return best;
    }

    public double getBestValidationLoss() {
        double best = Double.POSITIVE_INFINITY;
        for (EpochRecord e : epochs) {
            if (!Double.isNaN(e.validationLoss) && e.validationLoss < best) {
                best = e.validationLoss;
            }
        }
        return best;
    }

    public double getBestTrainAccuracy() {
        double best = Double.NEGATIVE_INFINITY;
        for (EpochRecord e : epochs) {
            if (!Double.isNaN(e.trainAccuracy) && e.trainAccuracy > best) {
                best = e.trainAccuracy;
            }
        }
        return best;
    }

    public double getBestValidationAccuracy() {
        double best = Double.NEGATIVE_INFINITY;
        for (EpochRecord e : epochs) {
            if (!Double.isNaN(e.validationAccuracy) && e.validationAccuracy > best) {
                best = e.validationAccuracy;
            }
        }
        return best;
    }

    public int getEpochWithBestValLoss() {
        int bestEpoch = -1;
        double best = Double.POSITIVE_INFINITY;
        for (EpochRecord e : epochs) {
            if (!Double.isNaN(e.validationLoss) && e.validationLoss < best) {
                best = e.validationLoss;
                bestEpoch = e.epochNumber;
            }
        }
        return bestEpoch;
    }

    public int getEpochWithBestValAccuracy() {
        int bestEpoch = -1;
        double best = Double.NEGATIVE_INFINITY;
        for (EpochRecord e : epochs) {
            if (!Double.isNaN(e.validationAccuracy) && e.validationAccuracy > best) {
                best = e.validationAccuracy;
                bestEpoch = e.epochNumber;
            }
        }
        return bestEpoch;
    }

    public Map<String, List<Double>> toHistoryMap() {
        Map<String, List<Double>> history = new LinkedHashMap<>();
        List<Double> trainLoss = new ArrayList<>();
        List<Double> valLoss = new ArrayList<>();
        List<Double> trainAcc = new ArrayList<>();
        List<Double> valAcc = new ArrayList<>();

        for (EpochRecord e : epochs) {
            trainLoss.add(e.trainLoss);
            if (!Double.isNaN(e.validationLoss)) valLoss.add(e.validationLoss);
            if (!Double.isNaN(e.trainAccuracy)) trainAcc.add(e.trainAccuracy);
            if (!Double.isNaN(e.validationAccuracy)) valAcc.add(e.validationAccuracy);
        }

        history.put("loss", trainLoss);
        if (!valLoss.isEmpty()) history.put("val_loss", valLoss);
        if (!trainAcc.isEmpty()) history.put("accuracy", trainAcc);
        if (!valAcc.isEmpty()) history.put("val_accuracy", valAcc);

        return history;
    }

    public void printSummary() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("EXECUTION MATRIX SUMMARY");
        System.out.println("=".repeat(80));
        
        if (modelName != null) {
            System.out.println("Model: " + modelName);
        }
        System.out.println("Total Epochs: " + epochs.size());
        System.out.println("Total Training Time: " + formatDuration(getTotalTrainingTimeMs()));
        System.out.println();
        
        if (!metadata.isEmpty()) {
            System.out.println("Metadata:");
            for (Map.Entry<String, Object> e : metadata.entrySet()) {
                System.out.println("  " + e.getKey() + ": " + e.getValue());
            }
            System.out.println();
        }

        System.out.println("Best Metrics:");
        System.out.println("  Best Training Loss:     " + String.format("%.6f", getBestTrainLoss()));
        if (!Double.isInfinite(getBestValidationLoss())) {
            System.out.println("  Best Validation Loss:   " + String.format("%.6f", getBestValidationLoss()) 
                    + " (epoch " + getEpochWithBestValLoss() + ")");
        }
        if (!Double.isInfinite(getBestTrainAccuracy())) {
            System.out.println("  Best Training Accuracy: " + String.format("%.4f (%.2f%%)", 
                    getBestTrainAccuracy(), getBestTrainAccuracy() * 100));
        }
        if (!Double.isInfinite(getBestValidationAccuracy())) {
            System.out.println("  Best Validation Acc:    " + String.format("%.4f (%.2f%%)", 
                    getBestValidationAccuracy(), getBestValidationAccuracy() * 100)
                    + " (epoch " + getEpochWithBestValAccuracy() + ")");
        }
        System.out.println();
    }

    public void printEpochTable() {
        System.out.println("\n" + "=".repeat(100));
        System.out.printf("%-6s | %-12s | %-12s | %-12s | %-12s | %-10s%n",
            "Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc", "Duration");
        System.out.println("-".repeat(100));

        for (EpochRecord e : epochs) {
            System.out.printf("%-6d | %-12.6f | %-12s | %-12s | %-12s | %-10s%n",
                e.epochNumber,
                e.trainLoss,
                Double.isNaN(e.validationLoss) ? "N/A" : String.format("%.6f", e.validationLoss),
                Double.isNaN(e.trainAccuracy) ? "N/A" : String.format("%.4f", e.trainAccuracy),
                Double.isNaN(e.validationAccuracy) ? "N/A" : String.format("%.4f", e.validationAccuracy),
                formatDuration(e.durationMs));
        }
        System.out.println("=".repeat(100));
    }

    public void exportToCSV(String filePath) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("Epoch,TrainLoss,ValidationLoss,TrainAccuracy,ValidationAccuracy,DurationMs");
            for (EpochRecord e : epochs) {
                writer.printf("%d,%.10f,%s,%s,%s,%d%n",
                    e.epochNumber,
                    e.trainLoss,
                    Double.isNaN(e.validationLoss) ? "" : String.format("%.10f", e.validationLoss),
                    Double.isNaN(e.trainAccuracy) ? "" : String.format("%.10f", e.trainAccuracy),
                    Double.isNaN(e.validationAccuracy) ? "" : String.format("%.10f", e.validationAccuracy),
                    e.durationMs);
            }
        }
        System.out.println("Execution matrix exported to: " + filePath);
    }

    public void save(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
        }
    }

    public static ExecutionMatrix load(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            return (ExecutionMatrix) ois.readObject();
        }
    }

    private String formatDuration(long ms) {
        if (ms < 1000) return ms + "ms";
        if (ms < 60000) return String.format("%.2fs", ms / 1000.0);
        long minutes = ms / 60000;
        long seconds = (ms % 60000) / 1000;
        return minutes + "m " + seconds + "s";
    }

    public static class EpochRecord implements Serializable {
        public int epochNumber;
        public double trainLoss;
        public double validationLoss;
        public double trainAccuracy;
        public double validationAccuracy;
        public long durationMs;
        public long startTime;
        public long endTime;
        public Map<String, Double> batchLosses;
        public Map<String, Double> extraMetrics;

        public EpochRecord(int epochNumber) {
            this.epochNumber = epochNumber;
            this.trainLoss = Double.NaN;
            this.validationLoss = Double.NaN;
            this.trainAccuracy = Double.NaN;
            this.validationAccuracy = Double.NaN;
            this.extraMetrics = new LinkedHashMap<>();
        }
    }
}
