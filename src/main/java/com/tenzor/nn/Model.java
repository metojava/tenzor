package com.tenzor.nn;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.nn.layers.*;
import com.tenzor.nn.losses.Loss;
import com.tenzor.nn.optimizers.Optimizer;
import java.util.*;

public class Model {
    private Layer inputLayer;
    private Layer outputLayer;
    private List<Layer> layers;
    private Loss loss;
    private Optimizer optimizer;
    private Map<String, List<Double>> history;

    public Model(Layer inputLayer, Layer outputLayer) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.layers = new ArrayList<>();
        this.history = new LinkedHashMap<>();
        buildLayerList();
    }

    private void buildLayerList() {
        layers.clear();
        Set<Layer> visited = new HashSet<>();
        Queue<Layer> queue = new LinkedList<>();
        queue.add(outputLayer);
        while (!queue.isEmpty()) {
            Layer l = queue.poll();
            if (!visited.contains(l)) {
                visited.add(l);
                layers.add(0, l);
            }
        }
    }

    public void compile(Loss loss, Optimizer optimizer) {
        this.loss = loss;
        this.optimizer = optimizer;
    }

    public Tensor forward(Tensor input) {
        Tensor current = inputLayer.forward(input);
        for (int i = 1; i < layers.size(); i++) {
            current = layers.get(i).forward(current);
        }
        return current;
    }

    public void fit(Tensor x, Tensor y, int epochs, int batchSize, double learningRate) {
        int n = x.shape().dim(0);
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int batches = 0;
            for (int i = 0; i < n; i += batchSize) {
                int end = Math.min(i + batchSize, n);
                Tensor xBatch = x.slice(0, i, end);
                Tensor yBatch = y.slice(0, i, end);
                Tensor pred = forward(xBatch);
                Tensor batchLoss = loss.forward(pred, yBatch);
                totalLoss += batchLoss.scalar();
                batches++;
            }
            double avgLoss = totalLoss / batches;
            history.computeIfAbsent("loss", k -> new ArrayList<>()).add(avgLoss);
            System.out.printf("Epoch %d/%d - loss: %.6f%n", epoch + 1, epochs, avgLoss);
        }
    }

    public Tensor predict(Tensor x) { return forward(x); }

    public List<Variable> getParameters() {
        List<Variable> params = new ArrayList<>();
        for (Layer l : layers) params.addAll(l.getParameters());
        return params;
    }

    public Map<String, List<Double>> getHistory() { return history; }
}
