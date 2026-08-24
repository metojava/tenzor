package com.tenzor.spring.boot.example;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.nn.Sequential;
import com.tenzor.spring.boot.TenzorTemplate;
import com.tenzor.spring.boot.service.DataPreparationService;
import com.tenzor.training.ExecutionMatrix;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@SpringBootApplication
public class TenzorDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(TenzorDemoApplication.class, args);
    }

    @Bean
    public CommandLineRunner demo(TenzorTemplate tenzorTemplate,
                                   DataPreparationService dataService) {
        return args -> {
            System.out.println("========================================");
            System.out.println("Tenzor Spring Boot Starter Demo");
            System.out.println("========================================");

            demo1_ClassificationModel(tenzorTemplate);
            demo2_TimeSeriesPrediction(tenzorTemplate, dataService);
            demo3_ModelSaveLoad(tenzorTemplate);

            System.out.println("\n========================================");
            System.out.println("Demo Complete!");
            System.out.println("========================================");
        };
    }

    private void demo1_ClassificationModel(TenzorTemplate tenzorTemplate) {
        System.out.println("\n--- Demo 1: Classification Model ---");

        TenzorTemplate.ModelConfig config = new TenzorTemplate.ModelConfig()
            .layer(TenzorTemplate.LayerConfig.dense(64, "relu"))
            .layer(TenzorTemplate.LayerConfig.dropout(0.2))
            .layer(TenzorTemplate.LayerConfig.dense(32, "relu"))
            .layer(TenzorTemplate.LayerConfig.dense(3, "softmax"))
            .loss("sparsecategoricalcrossentropy")
            .optimizer("adam")
            .learningRate(0.001);

        Sequential model = tenzorTemplate.createModel(config);
        System.out.println("Model created with " + model.getLayers().size() + " layers");

        Tensor xTrain = generateRandomData(100, 10);
        Tensor yTrain = generateRandomLabels(100, 3);

        System.out.println("Training model...");
        ExecutionMatrix history = tenzorTemplate.train(model, xTrain, yTrain, 20, 16, 0.001, 0.1);
        
        System.out.println("Best validation loss: " + history.getBestValidationLoss());
        System.out.println("Training completed in " + history.getTotalTrainingTimeMs() + "ms");
    }

    private void demo2_TimeSeriesPrediction(TenzorTemplate tenzorTemplate,
                                              DataPreparationService dataService) {
        System.out.println("\n--- Demo 2: Time Series Prediction ---");

        double[] prices = generateSineWave(200, 50, 1.0);
        System.out.println("Generated time series with " + prices.length + " data points");

        DataPreparationService.TimeSeriesResult result = 
            dataService.prepareTimeSeries(prices, 10, 1);

        System.out.println("xTrain shape: " + result.xTrain.shape());
        System.out.println("yTrain shape: " + result.yTrain.shape());

        TenzorTemplate.ModelConfig config = new TenzorTemplate.ModelConfig()
            .layer(TenzorTemplate.LayerConfig.lstm(32, true))
            .layer(TenzorTemplate.LayerConfig.dropout(0.1))
            .layer(TenzorTemplate.LayerConfig.lstm(16, false))
            .layer(TenzorTemplate.LayerConfig.dense(1, "linear"))
            .loss("mse")
            .optimizer("adam");

        Sequential model = tenzorTemplate.createModel(config);

        System.out.println("Training LSTM model...");
        ExecutionMatrix history = tenzorTemplate.train(
            model, result.xTrain, result.yTrain, 30, 16, 0.001, 0.1
        );

        System.out.println("Training complete. Final loss: " + history.getBestTrainLoss());
    }

    private void demo3_ModelSaveLoad(TenzorTemplate tenzorTemplate) {
        System.out.println("\n--- Demo 3: Model Save & Load ---");

        TenzorTemplate.ModelConfig config = new TenzorTemplate.ModelConfig()
            .layer(TenzorTemplate.LayerConfig.dense(16, "relu"))
            .layer(TenzorTemplate.LayerConfig.dense(1, "linear"))
            .loss("mse")
            .optimizer("adam");

        Sequential model = tenzorTemplate.createModel(config);
        System.out.println("Model created");

        tenzorTemplate.saveModel(model, "demo_model");
        System.out.println("Model saved");

        Sequential loadedModel = tenzorTemplate.loadModel("demo_model");
        System.out.println("Model loaded, layers: " + loadedModel.getLayers().size());
    }

    private Tensor generateRandomData(int samples, int features) {
        Random rand = new Random(42);
        double[][] data = new double[samples][features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = rand.nextDouble() * 2 - 1;
            }
        }
        return new Tensor(data);
    }

    private Tensor generateRandomLabels(int samples, int numClasses) {
        Random rand = new Random(42);
        double[] labels = new double[samples];
        for (int i = 0; i < samples; i++) {
            labels[i] = rand.nextInt(numClasses);
        }
        return new Tensor(labels);
    }

    private double[] generateSineWave(int length, int period, double amplitude) {
        double[] data = new double[length];
        for (int i = 0; i < length; i++) {
            data[i] = amplitude * Math.sin(2 * Math.PI * i / period) + 0.1 * (Math.random() - 0.5);
        }
        return data;
    }
}
