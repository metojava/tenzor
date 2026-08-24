package com.tenzor.examples;

import com.tenzor.core.Tensor;
import com.tenzor.datasets.FashionMNIST;
import com.tenzor.nn.Sequential;
import com.tenzor.nn.layers.*;
import com.tenzor.nn.losses.SparseCategoricalCrossentropy;
import com.tenzor.nn.optimizers.Adam;
import com.tenzor.metrics.Accuracy;
import com.tenzor.training.ExecutionMatrix;
import com.tenzor.visualization.ChartUtils;
import com.tenzor.callbacks.*;

import java.util.*;
import java.io.*;
import java.nio.file.*;

public class FashionMNISTExample {
    public static void main(String[] args) throws Exception {
        String outputDir = "output/fashion_mnist";
        Files.createDirectories(Paths.get(outputDir));

        System.out.println("=" .repeat(80));
        System.out.println("FASHION MNIST CLASSIFICATION");
        System.out.println("=".repeat(80));
        
        System.out.println("\nLoading Fashion MNIST dataset...");
        FashionMNIST dataset = new FashionMNIST();

        Tensor xTrain = dataset.getXTrain();
        Tensor yTrain = dataset.getYTrain();
        Tensor xTest = dataset.getXTest();
        Tensor yTest = dataset.getYTest();

        System.out.println("Train images: " + xTrain.shape());
        System.out.println("Train labels: " + yTrain.shape());
        System.out.println("Test images:  " + xTest.shape());
        System.out.println("Test labels:  " + yTest.shape());

        String[] classNames = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };

        System.out.println("\nBuilding neural network model...");
        
        Sequential model = new Sequential();
        model.add(new Dense(256, "relu"));
        model.add(new BatchNormalization());
        model.add(new Dropout(0.3));
        model.add(new Dense(128, "relu"));
        model.add(new BatchNormalization());
        model.add(new Dropout(0.3));
        model.add(new Dense(64, "relu"));
        model.add(new Dropout(0.2));
        model.add(new Dense(10, "softmax"));

        model.compile(new SparseCategoricalCrossentropy(), new Adam(0.001));

        EarlyStopping earlyStopping = new EarlyStopping("val_loss", 10, true);
        ModelCheckpoint checkpoint = new ModelCheckpoint(outputDir + "/best_model.json", "val_loss", true);
        ReduceLROnPlateau reduceLr = new ReduceLROnPlateau("val_loss", 0.5, 5, 1e-7);

        System.out.println("\nStarting training...");
        System.out.println();
        
        ExecutionMatrix execMatrix = model.fit(
            xTrain, yTrain, 50, 64, 0.001, 
            0.15,
            earlyStopping, checkpoint, reduceLr
        );

        execMatrix.printEpochTable();
        execMatrix.printSummary();

        String matrixCsvPath = outputDir + "/execution_matrix.csv";
        execMatrix.exportToCSV(matrixCsvPath);

        System.out.println("\nGenerating training charts...");
        ChartUtils.plotTrainingHistory(
            execMatrix.toHistoryMap(),
            "Fashion MNIST Training History",
            outputDir + "/training_history.png",
            1000, 600
        );

        System.out.println("\nEvaluating model on test set...");
        model.printEvaluation(xTest, yTest);

        Tensor predictions = model.predict(xTest);
        double accuracy = Accuracy.compute(yTest, predictions);
        System.out.println("\nFinal Test Accuracy: " + String.format("%.2f%%", accuracy * 100));

        int[] predLabels = argmax(predictions);
        double[] yTestData = yTest.getData();
        
        System.out.println("\nGenerating prediction analysis charts...");
        
        int sampleCount = Math.min(500, xTest.shape().dim(0));
        double[] actualSample = new double[sampleCount];
        double[] predSample = new double[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            actualSample[i] = yTestData[i];
            predSample[i] = predLabels[i];
        }
        
        double[] residuals = new double[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            residuals[i] = actualSample[i] - predSample[i];
        }
        ChartUtils.plotResiduals(actualSample, predSample, "Prediction Residuals", 
            outputDir + "/residuals.png");

        int[][] confusionMatrix = computeConfusionMatrix(yTestData, predLabels, 10);
        System.out.println("\nCONFUSION MATRIX:");
        printConfusionMatrix(confusionMatrix, classNames);

        double[] classAccuracy = new double[10];
        int[] classCounts = new int[10];
        for (int i = 0; i < yTestData.length; i++) {
            int actual = (int) yTestData[i];
            classCounts[actual]++;
            if (actual == predLabels[i]) {
                classAccuracy[actual]++;
            }
        }
        for (int i = 0; i < 10; i++) {
            if (classCounts[i] > 0) {
                classAccuracy[i] /= classCounts[i];
            }
        }
        
        ChartUtils.plotBarChart(classNames, classAccuracy, 
            "Per-Class Accuracy", "Class", "Accuracy",
            outputDir + "/class_accuracy.png");

        System.out.println("\nGenerating example predictions...");
        showPredictions(xTest, yTestData, predLabels, classNames, 10);

        String modelPath = outputDir + "/final_model.json";
        com.tenzor.io.ModelSaver.save(model, modelPath);
        System.out.println("\nFinal model saved to: " + modelPath);

        System.out.println("\n" + "=".repeat(80));
        System.out.println("TRAINING COMPLETE");
        System.out.println("=".repeat(80));
        System.out.println("\nOutput files saved to: " + new File(outputDir).getAbsolutePath());
        System.out.println("  - training_history.png");
        System.out.println("  - residuals.png");
        System.out.println("  - class_accuracy.png");
        System.out.println("  - execution_matrix.csv");
        System.out.println("  - final_model.json");
        System.out.println("  - best_model.json (checkpoint)");
    }

    private static int[] argmax(Tensor predictions) {
        int n = predictions.shape().dim(0);
        int classes = predictions.shape().dim(1);
        int[] result = new int[n];
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
        return result;
    }

    private static int[][] computeConfusionMatrix(double[] actual, int[] predicted, int numClasses) {
        int[][] matrix = new int[numClasses][numClasses];
        for (int i = 0; i < actual.length; i++) {
            int a = (int) actual[i];
            int p = predicted[i];
            if (a >= 0 && a < numClasses && p >= 0 && p < numClasses) {
                matrix[a][p]++;
            }
        }
        return matrix;
    }

    private static void printConfusionMatrix(int[][] matrix, String[] labels) {
        int n = matrix.length;
        System.out.print("\n          ");
        for (int i = 0; i < n; i++) {
            System.out.printf("%5d ", i);
        }
        System.out.println();
        System.out.println("          " + "-".repeat(6 * n));
        
        for (int i = 0; i < n; i++) {
            System.out.printf("%2d (%s) |", i, labels[i].substring(0, Math.min(5, labels[i].length())));
            for (int j = 0; j < n; j++) {
                System.out.printf("%5d ", matrix[i][j]);
            }
            System.out.println();
        }
    }

    private static void showPredictions(Tensor xTest, double[] actual, int[] predicted, 
                                         String[] classNames, int count) {
        System.out.println("\nSample Predictions:");
        System.out.println("-".repeat(70));
        System.out.printf("%-6s | %-15s | %-15s | %-6s%n", 
            "Index", "Actual", "Predicted", "Correct?");
        System.out.println("-".repeat(70));
        
        for (int i = 0; i < Math.min(count, actual.length); i++) {
            int a = (int) actual[i];
            int p = predicted[i];
            String correct = (a == p) ? "YES" : "NO";
            System.out.printf("%-6d | %-15s | %-15s | %-6s%n",
                i, classNames[a], classNames[p], correct);
        }
        System.out.println("-".repeat(70));
    }
}
