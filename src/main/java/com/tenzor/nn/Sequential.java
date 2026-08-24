package com.tenzor.nn;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.core.Shape;
import com.tenzor.nn.layers.*;
import com.tenzor.nn.losses.*;
import com.tenzor.nn.optimizers.Optimizer;
import com.tenzor.callbacks.Callback;
import com.tenzor.metrics.*;
import com.tenzor.training.ExecutionMatrix;

import java.util.*;

public class Sequential {
    private List<Layer> layers;
    private Loss loss;
    private Optimizer optimizer;
    private Map<String, List<Double>> history;
    private ExecutionMatrix executionMatrix;
    private boolean verbose;
    private List<String> metricsToTrack;

    public Sequential() {
        this.layers = new ArrayList<>();
        this.history = new LinkedHashMap<>();
        this.executionMatrix = new ExecutionMatrix();
        this.verbose = true;
        this.metricsToTrack = new ArrayList<>();
    }

    public void add(Layer layer) {
        layers.add(layer);
    }

    public void compile(Loss loss, Optimizer optimizer) {
        this.loss = loss;
        this.optimizer = optimizer;
    }

    public void compile(Loss loss, Optimizer optimizer, String... metrics) {
        this.loss = loss;
        this.optimizer = optimizer;
        this.metricsToTrack = Arrays.asList(metrics);
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (Layer layer : layers) {
            layer.resetState();
            current = layer.forward(current);
        }
        return current;
    }

    public ExecutionMatrix fit(Tensor x, Tensor y, int epochs, int batchSize, double learningRate) {
        return fit(x, y, epochs, batchSize, learningRate, 0.0, null, new Callback[0]);
    }

    public ExecutionMatrix fit(Tensor x, Tensor y, int epochs, int batchSize, double learningRate,
                               Callback... callbacks) {
        return fit(x, y, epochs, batchSize, learningRate, 0.0, null, callbacks);
    }

    public ExecutionMatrix fit(Tensor x, Tensor y, int epochs, int batchSize, double learningRate,
                               double validationSplit, Callback... callbacks) {
        return fit(x, y, epochs, batchSize, learningRate, validationSplit, null, callbacks);
    }

    public ExecutionMatrix fit(Tensor x, Tensor y, int epochs, int batchSize, double learningRate,
                               double validationSplit, Tensor[] validationData, Callback... callbacks) {
        List<Callback> cbList = new ArrayList<>(Arrays.asList(callbacks));

        int totalSamples = x.shape().dim(0);
        int valSize = (int) (totalSamples * validationSplit);
        int trainSize = totalSamples - valSize;

        Tensor xTrain, yTrain, xVal = null, yVal = null;
        
        if (validationData != null && validationData.length >= 2) {
            xTrain = x;
            yTrain = y;
            xVal = validationData[0];
            yVal = validationData[1];
        } else if (valSize > 0) {
            xTrain = x.slice(0, 0, trainSize);
            yTrain = y.slice(0, 0, trainSize);
            xVal = x.slice(0, trainSize, totalSamples);
            yVal = y.slice(0, trainSize, totalSamples);
        } else {
            xTrain = x;
            yTrain = y;
        }

        boolean isClassification = isClassificationProblem(yTrain, loss);

        executionMatrix = new ExecutionMatrix();
        executionMatrix.startTraining();
        executionMatrix.addMetadata("Epochs", epochs);
        executionMatrix.addMetadata("Batch Size", batchSize);
        executionMatrix.addMetadata("Learning Rate", learningRate);
        executionMatrix.addMetadata("Train Samples", xTrain.shape().dim(0));
        if (xVal != null) {
            executionMatrix.addMetadata("Validation Samples", xVal.shape().dim(0));
        }

        if (verbose) {
            printTrainingHeader(xTrain, xVal, epochs, batchSize);
        }

        for (Callback cb : cbList) cb.onTrainBegin(this);

        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStartTime = System.currentTimeMillis();
            int finalEpoch = epoch;
            
            for (Callback cb : cbList) cb.onEpochBegin(finalEpoch, this);

            double totalLoss = 0;
            int batches = 0;
            int correctTrain = 0;
            int totalTrain = 0;

            int[] indices = new int[trainSize];
            for (int i = 0; i < trainSize; i++) indices[i] = i;
            shuffleArray(indices);

            for (int i = 0; i < trainSize; i += batchSize) {
                int end = Math.min(i + batchSize, trainSize);
                int batchLen = end - i;

                int[] batchIndices = new int[batchLen];
                System.arraycopy(indices, i, batchIndices, 0, batchLen);

                Tensor xBatch = gatherRows(xTrain, batchIndices);
                Tensor yBatch = gatherRows(yTrain, batchIndices);

                Tensor predictions = forward(xBatch);
                Tensor batchLoss = loss.forward(predictions, yBatch);
                totalLoss += batchLoss.scalar();

                if (isClassification) {
                    int[] predLabels = argmax(predictions);
                    for (int j = 0; j < batchLen; j++) {
                        if (Math.abs(predLabels[j] - yBatch.get(j)) < 0.5) {
                            correctTrain++;
                        }
                        totalTrain++;
                    }
                }

                Tensor gradOutput = loss.backward(predictions, yBatch);
                backward(gradOutput, learningRate);

                batches++;
            }

            double trainLoss = totalLoss / batches;
            double trainAccuracy = isClassification && totalTrain > 0 ? (double) correctTrain / totalTrain : Double.NaN;
            double valLoss = Double.NaN;
            double valAccuracy = Double.NaN;

            if (xVal != null) {
                Tensor valPred = forward(xVal);
                Tensor valLossTensor = loss.forward(valPred, yVal);
                valLoss = valLossTensor.scalar();

                if (isClassification) {
                    int correct = 0;
                    int[] predLabels = argmax(valPred);
                    for (int j = 0; j < xVal.shape().dim(0); j++) {
                        if (Math.abs(predLabels[j] - yVal.get(j)) < 0.5) {
                            correct++;
                        }
                    }
                    valAccuracy = (double) correct / xVal.shape().dim(0);
                }
            }

            long epochDuration = System.currentTimeMillis() - epochStartTime;

            history.computeIfAbsent("loss", k -> new ArrayList<>()).add(trainLoss);
            if (!Double.isNaN(valLoss)) {
                history.computeIfAbsent("val_loss", k -> new ArrayList<>()).add(valLoss);
            }
            if (!Double.isNaN(trainAccuracy)) {
                history.computeIfAbsent("accuracy", k -> new ArrayList<>()).add(trainAccuracy);
            }
            if (!Double.isNaN(valAccuracy)) {
                history.computeIfAbsent("val_accuracy", k -> new ArrayList<>()).add(valAccuracy);
            }

            executionMatrix.addEpoch(epoch + 1, trainLoss, valLoss, trainAccuracy, valAccuracy, epochDuration);

            if (verbose) {
                printEpochProgress(epoch + 1, epochs, trainLoss, valLoss, trainAccuracy, valAccuracy, epochDuration);
            }

            for (Callback cb : cbList) cb.onEpochEnd(finalEpoch, this, trainLoss, valLoss);

            boolean shouldStop = false;
            for (Callback cb : cbList) {
                if (cb.shouldStop()) shouldStop = true;
            }
            if (shouldStop) {
                if (verbose) {
                    System.out.println("\nTraining stopped early by callback.");
                }
                break;
            }
        }

        executionMatrix.endTraining();
        for (Callback cb : cbList) cb.onTrainEnd(this);

        if (verbose) {
            executionMatrix.printSummary();
        }

        return executionMatrix;
    }

    private void backward(Tensor gradOutput, double learningRate) {
        Tensor grad = gradOutput;
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            if (layer instanceof Dense) {
                grad = ((Dense) layer).backward(grad, learningRate);
            } else if (layer instanceof Dropout) {
                grad = ((Dropout) layer).backward(grad);
            } else if (layer instanceof Activation) {
                continue;
            } else {
                if (layer instanceof Flatten) {
                    grad = grad.reshape(layer.getInput().shape().dims());
                }
            }
        }
    }

    public Tensor predict(Tensor x) {
        boolean wasTraining = true;
        for (Layer layer : layers) {
            if (layer instanceof Dropout) {
                ((Dropout) layer).setTraining(false);
            }
            if (layer instanceof BatchNormalization) {
                ((BatchNormalization) layer).setTraining(false);
            }
        }
        Tensor result = forward(x);
        for (Layer layer : layers) {
            if (layer instanceof Dropout) {
                ((Dropout) layer).setTraining(true);
            }
            if (layer instanceof BatchNormalization) {
                ((BatchNormalization) layer).setTraining(true);
            }
        }
        return result;
    }

    public Map<String, Double> evaluate(Tensor x, Tensor y) {
        Map<String, Double> results = new LinkedHashMap<>();
        
        Tensor predictions = predict(x);
        Tensor lossTensor = loss.forward(predictions, y);
        results.put("loss", lossTensor.scalar());

        boolean isClassification = isClassificationProblem(y, loss);
        if (isClassification) {
            double accuracy = Accuracy.compute(y, predictions);
            results.put("accuracy", accuracy);
            
            if (y.shape().numDims() == 1) {
                double precision = Precision.compute(y, predictions);
                double recall = Recall.compute(y, predictions);
                double f1 = F1Score.compute(y, predictions);
                results.put("precision", precision);
                results.put("recall", recall);
                results.put("f1", f1);
            }
        } else {
            double mse = com.tenzor.metrics.MeanSquaredError.compute(y, predictions);
            double mae = com.tenzor.metrics.MeanAbsoluteError.compute(y, predictions);
            results.put("mse", mse);
            results.put("mae", mae);
        }

        return results;
    }

    public void printEvaluation(Tensor x, Tensor y) {
        Map<String, Double> eval = evaluate(x, y);
        System.out.println("\n" + "=".repeat(50));
        System.out.println("EVALUATION RESULTS");
        System.out.println("=".repeat(50));
        for (Map.Entry<String, Double> e : eval.entrySet()) {
            String key = e.getKey();
            double val = e.getValue();
            if (key.equals("loss") || key.equals("mse") || key.equals("mae")) {
                System.out.printf("%-12s: %.6f%n", key, val);
            } else {
                System.out.printf("%-12s: %.4f (%.2f%%)%n", key, val, val * 100);
            }
        }
        System.out.println("=".repeat(50));
    }

    public List<Variable> getParameters() {
        List<Variable> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.getParameters());
        }
        return params;
    }

    public List<Layer> getLayers() { return layers; }

    public Layer getLayer(int index) { return layers.get(index); }

    public Map<String, List<Double>> getHistory() { return history; }

    public ExecutionMatrix getExecutionMatrix() { return executionMatrix; }

    public Loss getLoss() { return loss; }

    public Optimizer getOptimizer() { return optimizer; }

    private boolean isClassificationProblem(Tensor y, Loss loss) {
        if (loss instanceof CategoricalCrossentropy || 
            loss instanceof SparseCategoricalCrossentropy ||
            loss instanceof Hinge) {
            return true;
        }
        if (loss instanceof BinaryCrossentropy) {
            double[] data = y.getData();
            for (double v : data) {
                if (v == 0.0 || v == 1.0) return true;
            }
        }
        return false;
    }

    private int[] argmax(Tensor predictions) {
        int n = predictions.shape().dim(0);
        int[] result = new int[n];
        
        if (predictions.shape().numDims() == 2) {
            int classes = predictions.shape().dim(1);
            for (int i = 0; i < n; i++) {
                int maxIdx = 0;
                double maxVal = predictions.get(i, 0);
                for (int j = 1; j < classes; j++) {
                    double val = predictions.get(i, j);
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = j;
                    }
                }
                result[i] = maxIdx;
            }
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = predictions.get(i) > 0.5 ? 1 : 0;
            }
        }
        return result;
    }

    private void printTrainingHeader(Tensor xTrain, Tensor xVal, int epochs, int batchSize) {
        System.out.println("=".repeat(80));
        System.out.println("MODEL TRAINING");
        System.out.println("=".repeat(80));
        System.out.println("Train samples:      " + xTrain.shape().dim(0));
        System.out.println("Validation samples: " + (xVal != null ? xVal.shape().dim(0) : "None"));
        System.out.println("Epochs:             " + epochs);
        System.out.println("Batch size:         " + batchSize);
        System.out.println("=".repeat(80));
        System.out.println();
        System.out.printf("%-6s | %-12s | %-12s | %-12s | %-12s | %-10s%n",
            "Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc", "Duration");
        System.out.println("-".repeat(100));
    }

    private void printEpochProgress(int epoch, int epochs, double trainLoss, double valLoss,
                                      double trainAcc, double valAcc, long durationMs) {
        String epochStr = String.format("%d/%d", epoch, epochs);
        String durStr = String.format("%.2fs", durationMs / 1000.0);
        
        System.out.printf("%-6s | %-12.6f | %-12s | %-12s | %-12s | %-10s%n",
            epochStr,
            trainLoss,
            Double.isNaN(valLoss) ? "N/A" : String.format("%.6f", valLoss),
            Double.isNaN(trainAcc) ? "N/A" : String.format("%.4f", trainAcc),
            Double.isNaN(valAcc) ? "N/A" : String.format("%.4f", valAcc),
            durStr);
    }

    private void shuffleArray(int[] arr) {
        Random rng = new Random();
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    private Tensor gatherRows(Tensor source, int[] indices) {
        int[] dims = source.shape().dims();
        int[] newDims = new int[dims.length];
        newDims[0] = indices.length;
        System.arraycopy(dims, 1, newDims, 1, dims.length - 1);

        int rowSize = 1;
        for (int i = 1; i < dims.length; i++) rowSize *= dims[i];

        double[] srcData = source.getData();
        double[] resultData = new double[indices.length * rowSize];

        for (int i = 0; i < indices.length; i++) {
            System.arraycopy(srcData, indices[i] * rowSize, resultData, i * rowSize, rowSize);
        }

        return new Tensor(resultData, new Shape(newDims));
    }
}
