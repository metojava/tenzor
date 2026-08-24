package com.tenzor.spring.boot;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.nn.Sequential;
import com.tenzor.nn.layers.*;
import com.tenzor.nn.losses.*;
import com.tenzor.nn.optimizers.*;
import com.tenzor.training.ExecutionMatrix;
import com.tenzor.visualization.ChartUtils;
import com.tenzor.callbacks.*;
import com.tenzor.io.ModelSaver;
import com.tenzor.io.ModelLoader;
import com.tenzor.spring.boot.autoconfigure.TenzorProperties;
import org.springframework.util.FileCopyUtils;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class TenzorTemplate {

    private final TenzorProperties properties;
    private Sequential defaultModel;

    public TenzorTemplate(TenzorProperties properties) {
        this.properties = properties;
        initializeDirectories();
    }

    private void initializeDirectories() {
        try {
            Files.createDirectories(Paths.get(properties.getModel().getSavePath()));
            Files.createDirectories(Paths.get(properties.getTraining().getCheckpointPath()));
            if (properties.getVisualization().isEnabled()) {
                Files.createDirectories(Paths.get(properties.getVisualization().getOutputPath()));
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize Tenzor directories", e);
        }
    }

    public Sequential createModel(ModelConfig config) {
        Sequential model = new Sequential();
        
        for (LayerConfig layer : config.getLayers()) {
            switch (layer.getType().toLowerCase()) {
                case "dense":
                    model.add(new Dense(layer.getUnits(), layer.getActivation()));
                    break;
                case "dropout":
                    model.add(new Dropout(layer.getRate()));
                    break;
                case "lstm":
                    model.add(new LSTM(layer.getUnits(), layer.isReturnSequences()));
                    break;
                case "gru":
                    model.add(new GRU(layer.getUnits(), layer.isReturnSequences()));
                    break;
                case "simplernn":
                    model.add(new SimpleRNN(layer.getUnits(), layer.isReturnSequences()));
                    break;
                case "flatten":
                    model.add(new Flatten());
                    break;
                case "batchnormalization":
                    model.add(new BatchNormalization());
                    break;
                case "conv2d":
                    model.add(new Conv2D(layer.getFilters(), layer.getKernelSize(), layer.getActivation()));
                    break;
                case "maxpooling2d":
                    model.add(new MaxPooling2D(layer.getPoolSize()));
                    break;
                case "averagepooling2d":
                    model.add(new AveragePooling2D(layer.getPoolSize()));
                    break;
                case "embedding":
                    model.add(new Embedding(layer.getVocabSize(), layer.getEmbeddingDim()));
                    break;
                case "activation":
                    model.add(new Activation(layer.getActivation()));
                    break;
                default:
                    throw new IllegalArgumentException("Unknown layer type: " + layer.getType());
            }
        }
        
        model.compile(getLoss(config.getLoss()), getOptimizer(config.getOptimizer(), config.getLearningRate()));
        
        return model;
    }

    public ExecutionMatrix train(Sequential model, Tensor xTrain, Tensor yTrain) {
        return train(model, xTrain, yTrain, 
            properties.getTraining().getEpochs(),
            properties.getTraining().getBatchSize(),
            properties.getTraining().getLearningRate(),
            properties.getTraining().getValidationSplit()
        );
    }

    public ExecutionMatrix train(Sequential model, Tensor xTrain, Tensor yTrain,
                                   int epochs, int batchSize, double learningRate, double validationSplit) {
        
        List<Callback> callbacks = new ArrayList<>();
        
        if (properties.getTraining().isEarlyStopping()) {
            callbacks.add(new EarlyStopping(
                "val_loss",
                properties.getTraining().getEarlyStoppingPatience(),
                true
            ));
        }
        
        if (properties.getTraining().isCheckpointing()) {
            callbacks.add(new ModelCheckpoint(
                properties.getTraining().getCheckpointPath() + "best_model.json",
                "val_loss",
                true
            ));
        }
        
        ExecutionMatrix execMatrix = model.fit(
            xTrain, yTrain,
            epochs,
            batchSize,
            learningRate,
            validationSplit,
            callbacks.toArray(new Callback[0])
        );
        
        if (properties.getVisualization().isEnabled()) {
            try {
                ChartUtils.plotTrainingHistory(
                    execMatrix.toHistoryMap(),
                    "Training History",
                    properties.getVisualization().getOutputPath() + "training_history.png",
                    properties.getVisualization().getChartWidth(),
                    properties.getVisualization().getChartHeight()
                );
            } catch (Exception e) {
                System.err.println("Warning: Failed to generate training chart: " + e.getMessage());
            }
        }
        
        return execMatrix;
    }

    public Tensor predict(Sequential model, Tensor input) {
        return model.predict(input);
    }

    public Map<String, Double> evaluate(Sequential model, Tensor xTest, Tensor yTest) {
        return model.evaluate(xTest, yTest);
    }

    public void printEvaluation(Sequential model, Tensor xTest, Tensor yTest) {
        model.printEvaluation(xTest, yTest);
    }

    public void saveModel(Sequential model, String name) {
        try {
            String path = properties.getModel().getSavePath() + name + ".json";
            ModelSaver.save(model, path);
            System.out.println("Model saved to: " + path);
        } catch (Exception e) {
            throw new RuntimeException("Failed to save model", e);
        }
    }

    public Sequential loadModel(String name) {
        try {
            String path = properties.getModel().getSavePath() + name + ".json";
            if (!Files.exists(Paths.get(path))) {
                if (properties.getModel().getLoadPath() != null) {
                    path = properties.getModel().getLoadPath();
                }
            }
            return ModelLoader.load(path);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load model", e);
        }
    }

    public Sequential loadModelFromPath(String path) {
        try {
            return ModelLoader.load(path);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load model from path: " + path, e);
        }
    }

    public Tensor createTensor(double[] data) {
        return new Tensor(data);
    }

    public Tensor createTensor(double[][] data) {
        return new Tensor(data);
    }

    public Tensor createTensor(double[] flatData, Shape shape) {
        return new Tensor(flatData, shape);
    }

    public Tensor createTensor(List<Double> data) {
        double[] arr = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            arr[i] = data.get(i);
        }
        return new Tensor(arr);
    }

    public Tensor zeros(int... dims) {
        return Tensor.zeros(new Shape(dims));
    }

    public Tensor ones(int... dims) {
        return Tensor.ones(new Shape(dims));
    }

    public Tensor random(int... dims) {
        return Tensor.random(new Shape(dims), -0.1, 0.1);
    }

    public Tensor random(double min, double max, int... dims) {
        return Tensor.random(new Shape(dims), min, max);
    }

    public void plotPredictions(double[] actual, double[] predicted, String title) {
        if (!properties.getVisualization().isEnabled()) {
            return;
        }
        try {
            ChartUtils.plotPredictions(
                actual, predicted,
                title,
                properties.getVisualization().getOutputPath() + "predictions.png",
                properties.getVisualization().getChartWidth(),
                properties.getVisualization().getChartHeight()
            );
        } catch (Exception e) {
            System.err.println("Warning: Failed to generate predictions chart: " + e.getMessage());
        }
    }

    public void plotTimeSeries(double[] data, String title) {
        if (!properties.getVisualization().isEnabled()) {
            return;
        }
        try {
            ChartUtils.plotTimeSeries(
                data,
                title,
                properties.getVisualization().getOutputPath() + "timeseries.png"
            );
        } catch (Exception e) {
            System.err.println("Warning: Failed to generate time series chart: " + e.getMessage());
        }
    }

    public void plotBarChart(String[] names, double[] values, String title, String xLabel, String yLabel) {
        if (!properties.getVisualization().isEnabled()) {
            return;
        }
        try {
            ChartUtils.plotBarChart(
                names, values,
                title, xLabel, yLabel,
                properties.getVisualization().getOutputPath() + "barchart.png",
                properties.getVisualization().getChartWidth(),
                properties.getVisualization().getChartHeight()
            );
        } catch (Exception e) {
            System.err.println("Warning: Failed to generate bar chart: " + e.getMessage());
        }
    }

    private Loss getLoss(String lossName) {
        if (lossName == null) {
            return new MeanSquaredError();
        }
        switch (lossName.toLowerCase().replace("_", "")) {
            case "mse":
            case "meansquarederror":
                return new MeanSquaredError();
            case "mae":
            case "meanabsoluteerror":
                return new MeanAbsoluteError();
            case "binarycrossentropy":
                return new BinaryCrossentropy();
            case "categoricalcrossentropy":
                return new CategoricalCrossentropy();
            case "sparsecategoricalcrossentropy":
                return new SparseCategoricalCrossentropy();
            case "huber":
                return new Huber();
            case "hinge":
                return new Hinge();
            default:
                throw new IllegalArgumentException("Unknown loss function: " + lossName);
        }
    }

    private Optimizer getOptimizer(String optimizerName, double learningRate) {
        if (optimizerName == null) {
            return new Adam(learningRate);
        }
        switch (optimizerName.toLowerCase()) {
            case "adam":
                return new Adam(learningRate);
            case "sgd":
                return new SGD(learningRate, 0.9);
            case "rmsprop":
                return new RMSprop(learningRate);
            case "adagrad":
                return new Adagrad(learningRate);
            case "adadelta":
                return new Adadelta();
            default:
                throw new IllegalArgumentException("Unknown optimizer: " + optimizerName);
        }
    }

    public static class ModelConfig {
        private List<LayerConfig> layers = new ArrayList<>();
        private String loss = "mse";
        private String optimizer = "adam";
        private double learningRate = 0.001;

        public ModelConfig layer(LayerConfig layer) {
            this.layers.add(layer);
            return this;
        }

        public ModelConfig loss(String loss) {
            this.loss = loss;
            return this;
        }

        public ModelConfig optimizer(String optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public ModelConfig learningRate(double lr) {
            this.learningRate = lr;
            return this;
        }

        public List<LayerConfig> getLayers() {
            return layers;
        }

        public String getLoss() {
            return loss;
        }

        public String getOptimizer() {
            return optimizer;
        }

        public double getLearningRate() {
            return learningRate;
        }
    }

    public static class LayerConfig {
        private String type;
        private int units;
        private String activation;
        private double rate;
        private boolean returnSequences;
        private int filters;
        private int kernelSize;
        private int poolSize;
        private int vocabSize;
        private int embeddingDim;

        public static LayerConfig dense(int units, String activation) {
            LayerConfig config = new LayerConfig();
            config.type = "dense";
            config.units = units;
            config.activation = activation;
            return config;
        }

        public static LayerConfig dropout(double rate) {
            LayerConfig config = new LayerConfig();
            config.type = "dropout";
            config.rate = rate;
            return config;
        }

        public static LayerConfig lstm(int units, boolean returnSequences) {
            LayerConfig config = new LayerConfig();
            config.type = "lstm";
            config.units = units;
            config.returnSequences = returnSequences;
            return config;
        }

        public static LayerConfig gru(int units, boolean returnSequences) {
            LayerConfig config = new LayerConfig();
            config.type = "gru";
            config.units = units;
            config.returnSequences = returnSequences;
            return config;
        }

        public static LayerConfig flatten() {
            LayerConfig config = new LayerConfig();
            config.type = "flatten";
            return config;
        }

        public static LayerConfig batchNormalization() {
            LayerConfig config = new LayerConfig();
            config.type = "batchnormalization";
            return config;
        }

        public static LayerConfig conv2d(int filters, int kernelSize, String activation) {
            LayerConfig config = new LayerConfig();
            config.type = "conv2d";
            config.filters = filters;
            config.kernelSize = kernelSize;
            config.activation = activation;
            return config;
        }

        public static LayerConfig maxPooling2d(int poolSize) {
            LayerConfig config = new LayerConfig();
            config.type = "maxpooling2d";
            config.poolSize = poolSize;
            return config;
        }

        public static LayerConfig embedding(int vocabSize, int embeddingDim) {
            LayerConfig config = new LayerConfig();
            config.type = "embedding";
            config.vocabSize = vocabSize;
            config.embeddingDim = embeddingDim;
            return config;
        }

        public String getType() {
            return type;
        }

        public int getUnits() {
            return units;
        }

        public String getActivation() {
            return activation;
        }

        public double getRate() {
            return rate;
        }

        public boolean isReturnSequences() {
            return returnSequences;
        }

        public int getFilters() {
            return filters;
        }

        public int getKernelSize() {
            return kernelSize;
        }

        public int getPoolSize() {
            return poolSize;
        }

        public int getVocabSize() {
            return vocabSize;
        }

        public int getEmbeddingDim() {
            return embeddingDim;
        }
    }

    public TenzorProperties getProperties() {
        return properties;
    }
}
