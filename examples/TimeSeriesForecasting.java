package com.tenzor.examples;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.nn.Sequential;
import com.tenzor.nn.layers.*;
import com.tenzor.nn.losses.MeanSquaredError;
import com.tenzor.nn.optimizers.Adam;
import com.tenzor.metrics.*;
import com.tenzor.training.ExecutionMatrix;
import com.tenzor.visualization.ChartUtils;
import com.tenzor.timeseries.*;
import com.tenzor.callbacks.*;

import java.util.*;
import java.io.*;
import java.nio.file.*;

public class TimeSeriesForecasting {
    public static void main(String[] args) throws Exception {
        String outputDir = "output/time_series";
        Files.createDirectories(Paths.get(outputDir));

        System.out.println("=".repeat(80));
        System.out.println("TIME SERIES FORECASTING WITH LSTM");
        System.out.println("=".repeat(80));

        System.out.println("\nGenerating synthetic time series data...");
        
        int totalPoints = 1200;
        double[] sineWave = SequenceGenerator.generateSineWave(totalPoints, 50, 1.0, 0);
        double[] trend = SequenceGenerator.generateLinearTrend(totalPoints, 0.0, 0.002);
        double[] noise = SequenceGenerator.generateRandomWalk(totalPoints, 0, 0.1);
        
        double[] combined = new double[totalPoints];
        for (int i = 0; i < totalPoints; i++) {
            combined[i] = sineWave[i] + trend[i] + 0.5 * noise[i];
        }

        ChartUtils.plotTimeSeries(combined, "Input Time Series (Sine + Trend + Noise)", 
            outputDir + "/input_series.png");
        
        ChartUtils.plotTimeSeries(sineWave, "Sine Wave Component", 
            outputDir + "/sine_component.png");
        ChartUtils.plotTimeSeries(trend, "Trend Component", 
            outputDir + "/trend_component.png");

        System.out.println("Preprocessing data...");
        TimeSeriesPreprocessor preprocessor = new TimeSeriesPreprocessor();
        double[] normalized = preprocessor.normalizeMinMax(combined, -1.0, 1.0);

        int sequenceLength = 30;
        int predictionHorizon = 1;
        
        System.out.println("Creating sequences (sequence length: " + sequenceLength + ")...");
        TimeSeriesDataset dataset = new TimeSeriesDataset(normalized, sequenceLength, predictionHorizon);
        Tensor[] split = dataset.trainTestSplit(0.8);
        
        Tensor xTrain = split[0].reshape(split[0].shape().dim(0), sequenceLength, 1);
        Tensor yTrain = split[1];
        Tensor xTest = split[2].reshape(split[2].shape().dim(0), sequenceLength, 1);
        Tensor yTest = split[3];

        System.out.println("Train X: " + xTrain.shape() + ", Train Y: " + yTrain.shape());
        System.out.println("Test X:  " + xTest.shape() + ", Test Y:  " + yTest.shape());

        System.out.println("\nBuilding LSTM model...");
        
        Sequential model = new Sequential();
        model.add(new LSTM(64, true));
        model.add(new Dropout(0.2));
        model.add(new LSTM(32, false));
        model.add(new Dropout(0.1));
        model.add(new Dense(16, "relu"));
        model.add(new Dense(1, "linear"));

        model.compile(new MeanSquaredError(), new Adam(0.001));

        EarlyStopping earlyStopping = new EarlyStopping("val_loss", 15, true);
        ModelCheckpoint checkpoint = new ModelCheckpoint(outputDir + "/best_model.json", "val_loss", true);
        ReduceLROnPlateau reduceLr = new ReduceLROnPlateau("val_loss", 0.7, 5, 1e-8);

        System.out.println("\nTraining model...");
        System.out.println();
        
        ExecutionMatrix execMatrix = model.fit(
            xTrain, yTrain, 100, 32, 0.001, 
            0.1,
            earlyStopping, checkpoint, reduceLr
        );

        execMatrix.printEpochTable();
        execMatrix.printSummary();

        execMatrix.exportToCSV(outputDir + "/execution_matrix.csv");

        System.out.println("\nGenerating training charts...");
        ChartUtils.plotTrainingHistory(
            execMatrix.toHistoryMap(),
            "Time Series Training History",
            outputDir + "/training_history.png",
            1000, 600
        );

        System.out.println("\nEvaluating model...");
        model.printEvaluation(xTest, yTest);

        Tensor predictions = model.predict(xTest);
        
        double mse = MeanSquaredError.compute(yTest, predictions);
        double mae = MeanAbsoluteError.compute(yTest, predictions);
        double mape = MeanAbsolutePercentageError.compute(yTest, predictions);
        double r2 = R2Score.compute(yTest, predictions);

        System.out.println("\n" + "=".repeat(50));
        System.out.println("REGRESSION METRICS");
        System.out.println("=".repeat(50));
        System.out.printf("MSE:  %.6f%n", mse);
        System.out.printf("RMSE: %.6f%n", Math.sqrt(mse));
        System.out.printf("MAE:  %.6f%n", mae);
        System.out.printf("MAPE: %.4f%%%n", mape);
        System.out.printf("R²:   %.4f%n", r2);
        System.out.println("=".repeat(50));

        double[] yTestDenorm = preprocessor.denormalizeMinMax(yTest.getData(), -1.0, 1.0, 
            Arrays.stream(combined).min().getAsDouble(),
            Arrays.stream(combined).max().getAsDouble());
            
        double[] predDenorm = preprocessor.denormalizeMinMax(predictions.getData(), -1.0, 1.0,
            Arrays.stream(combined).min().getAsDouble(),
            Arrays.stream(combined).max().getAsDouble());

        System.out.println("\nGenerating prediction charts...");
        
        int plotStart = Math.max(0, yTestDenorm.length - 200);
        double[] actualPlot = Arrays.copyOfRange(yTestDenorm, plotStart, yTestDenorm.length);
        double[] predPlot = Arrays.copyOfRange(predDenorm, plotStart, predDenorm.length);
        
        ChartUtils.plotPredictions(actualPlot, predPlot, 
            "Time Series Predictions vs Actual (Last 200 points)",
            outputDir + "/predictions.png",
            1200, 600
        );

        double[] errors = new double[yTestDenorm.length];
        for (int i = 0; i < errors.length; i++) {
            errors[i] = yTestDenorm[i] - predDenorm[i];
        }
        ChartUtils.plotTimeSeries(errors, "Prediction Errors", 
            outputDir + "/errors.png");
        
        ChartUtils.plotHistogram(errors, "Error Distribution", "Error", 
            outputDir + "/error_histogram.png", 30);

        System.out.println("\nGenerating multi-step forecast example...");
        double[] multiStepForecast = generateMultiStepForecast(model, xTest, 10, sequenceLength);
        
        List<double[]> multiStepData = new ArrayList<>();
        List<String> multiStepLabels = new ArrayList<>();
        
        int forecastStart = 50;
        double[] actualForecast = Arrays.copyOfRange(yTestDenorm, forecastStart, forecastStart + 50);
        multiStepData.add(actualForecast);
        multiStepLabels.add("Actual");
        
        double[] forecastPlot = new double[50];
        System.arraycopy(predDenorm, forecastStart, forecastPlot, 0, Math.min(10, 50 - forecastStart));
        for (int i = 0; i < multiStepForecast.length && i < 50; i++) {
            if (forecastStart + 10 + i < 50) {
                forecastPlot[10 + i] = multiStepForecast[i];
            }
        }
        
        System.out.println("\nSaving model and results...");
        com.tenzor.io.ModelSaver.save(model, outputDir + "/final_model.json");

        System.out.println("\n" + "=".repeat(80));
        System.out.println("FORECASTING COMPLETE");
        System.out.println("=".repeat(80));
        System.out.println("\nOutput files saved to: " + new File(outputDir).getAbsolutePath());
    }

    private static double[] generateMultiStepForecast(Sequential model, Tensor initialInput, 
                                                        int steps, int seqLength) {
        double[] forecast = new double[steps];
        Tensor current = initialInput.slice(0, 0, 1);
        
        for (int i = 0; i < steps; i++) {
            Tensor pred = model.predict(current);
            forecast[i] = pred.get(0);
            
            double[] currentData = current.getData();
            double[] newData = new double[currentData.length];
            System.arraycopy(currentData, 1, newData, 0, currentData.length - 1);
            newData[newData.length - 1] = forecast[i];
            current = new Tensor(newData, new Shape(1, seqLength, 1));
        }
        
        return forecast;
    }
}
