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

public class StockPricePrediction {
    public static void main(String[] args) throws Exception {
        String outputDir = "output/stock_prediction";
        Files.createDirectories(Paths.get(outputDir));

        System.out.println("=".repeat(80));
        System.out.println("STOCK PRICE PREDICTION WITH TECHNICAL INDICATORS");
        System.out.println("=".repeat(80));

        double[] prices;
        
        if (args.length >= 1) {
            System.out.println("\nLoading stock data from: " + args[0]);
            StockDataLoader loader = new StockDataLoader();
            prices = loader.loadFromCSV(args[0]);
        } else {
            System.out.println("\nGenerating synthetic stock data for demonstration...");
            prices = generateSyntheticStockData(800);
        }

        System.out.println("Loaded " + prices.length + " price points");

        System.out.println("\nComputing technical indicators...");
        double[] sma5 = TechnicalIndicators.simpleMovingAverage(prices, 5);
        double[] sma20 = TechnicalIndicators.simpleMovingAverage(prices, 20);
        double[] ema12 = TechnicalIndicators.exponentialMovingAverage(prices, 12);
        double[] ema26 = TechnicalIndicators.exponentialMovingAverage(prices, 26);
        double[] macd = TechnicalIndicators.macd(prices, 12, 26, 9);
        double[] rsi = TechnicalIndicators.rsi(prices, 14);
        double[][] bollinger = TechnicalIndicators.bollingerBands(prices, 20, 2);
        double[] upperBand = bollinger[0];
        double[] middleBand = bollinger[1];
        double[] lowerBand = bollinger[2];
        double[] obv = TechnicalIndicators.onBalanceVolume(prices, generateVolume(prices.length));

        System.out.println("  - SMA 5, SMA 20");
        System.out.println("  - EMA 12, EMA 26");
        System.out.println("  - MACD (12, 26, 9)");
        System.out.println("  - RSI (14)");
        System.out.println("  - Bollinger Bands (20, 2)");
        System.out.println("  - On-Balance Volume");

        System.out.println("\nGenerating technical indicator charts...");
        
        List<double[]> priceData = new ArrayList<>();
        List<String> priceLabels = new ArrayList<>();
        priceData.add(Arrays.copyOfRange(prices, 100, prices.length));
        priceData.add(Arrays.copyOfRange(sma5, 100, prices.length));
        priceData.add(Arrays.copyOfRange(sma20, 100, prices.length));
        priceLabels.add("Price");
        priceLabels.add("SMA 5");
        priceLabels.add("SMA 20");
        
        ChartUtils.plotMultipleMetrics(priceData, priceLabels,
            "Stock Price with Moving Averages", "Time", "Price",
            outputDir + "/price_ma.png");

        List<double[]> bbData = new ArrayList<>();
        List<String> bbLabels = new ArrayList<>();
        bbData.add(Arrays.copyOfRange(prices, 100, prices.length));
        bbData.add(Arrays.copyOfRange(upperBand, 100, prices.length));
        bbData.add(Arrays.copyOfRange(middleBand, 100, prices.length));
        bbData.add(Arrays.copyOfRange(lowerBand, 100, prices.length));
        bbLabels.add("Price");
        bbLabels.add("Upper Band");
        bbLabels.add("Middle Band");
        bbLabels.add("Lower Band");
        
        ChartUtils.plotMultipleMetrics(bbData, bbLabels,
            "Bollinger Bands", "Time", "Price",
            outputDir + "/bollinger.png");

        ChartUtils.plotTimeSeries(Arrays.copyOfRange(rsi, 100, prices.length), 
            "RSI (14)", outputDir + "/rsi.png");
        
        ChartUtils.plotTimeSeries(Arrays.copyOfRange(macd, 100, prices.length), 
            "MACD", outputDir + "/macd.png");

        System.out.println("\nPreparing features and labels...");
        
        int n = prices.length;
        int featureCount = 10;
        double[][] features = new double[n][featureCount];
        
        for (int i = 0; i < n; i++) {
            features[i][0] = prices[i];
            features[i][1] = sma5[i];
            features[i][2] = sma20[i];
            features[i][3] = ema12[i];
            features[i][4] = rsi[i];
            features[i][5] = macd[i];
            features[i][6] = upperBand[i];
            features[i][7] = lowerBand[i];
            features[i][8] = (prices[i] - sma20[i]) / (sma20[i] + 1e-8);
            features[i][9] = obv[i];
        }

        TimeSeriesPreprocessor preprocessor = new TimeSeriesPreprocessor();
        double[][] normalizedFeatures = new double[n][featureCount];
        
        for (int j = 0; j < featureCount; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = features[i][j];
            col = preprocessor.normalizeZScore(col);
            for (int i = 0; i < n; i++) normalizedFeatures[i][j] = col[i];
        }

        int sequenceLength = 15;
        int predictionHorizon = 1;
        
        System.out.println("Creating sequences with length: " + sequenceLength);
        
        int samples = n - sequenceLength;
        double[][] xData = new double[samples][sequenceLength * featureCount];
        double[] yData = new double[samples];
        
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                for (int k = 0; k < featureCount; k++) {
                    xData[i][j * featureCount + k] = normalizedFeatures[i + j][k];
                }
            }
            yData[i] = prices[i + sequenceLength];
        }

        int trainSize = (int) (samples * 0.8);
        
        Tensor x = new Tensor(xData);
        Tensor y = new Tensor(yData, new Shape(samples));
        
        Tensor xTrain = x.slice(0, 0, trainSize);
        Tensor yTrain = y.slice(0, 0, trainSize);
        Tensor xTest = x.slice(0, trainSize, samples);
        Tensor yTest = y.slice(0, trainSize, samples);

        xTrain = xTrain.reshape(trainSize, sequenceLength, featureCount);
        xTest = xTest.reshape(xTest.shape().dim(0), sequenceLength, featureCount);

        System.out.println("Train X: " + xTrain.shape() + ", Train Y: " + yTrain.shape());
        System.out.println("Test X:  " + xTest.shape() + ", Test Y:  " + yTest.shape());

        System.out.println("\nBuilding stock prediction model...");
        
        Sequential model = new Sequential();
        model.add(new GRU(100, true));
        model.add(new Dropout(0.25));
        model.add(new GRU(50, true));
        model.add(new Dropout(0.2));
        model.add(new LSTM(25, false));
        model.add(new Dense(32, "relu"));
        model.add(new Dropout(0.1));
        model.add(new Dense(1, "linear"));

        model.compile(new MeanSquaredError(), new Adam(0.001));

        EarlyStopping earlyStopping = new EarlyStopping("val_loss", 20, true);
        ModelCheckpoint checkpoint = new ModelCheckpoint(outputDir + "/best_model.json", "val_loss", true);
        ReduceLROnPlateau reduceLr = new ReduceLROnPlateau("val_loss", 0.5, 8, 1e-8);

        System.out.println("\nTraining model...");
        System.out.println();
        
        ExecutionMatrix execMatrix = model.fit(
            xTrain, yTrain, 150, 64, 0.001, 
            0.1,
            earlyStopping, checkpoint, reduceLr
        );

        execMatrix.printEpochTable();
        execMatrix.printSummary();
        execMatrix.exportToCSV(outputDir + "/execution_matrix.csv");

        System.out.println("\nGenerating training charts...");
        ChartUtils.plotTrainingHistory(
            execMatrix.toHistoryMap(),
            "Stock Model Training History",
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
        System.out.println("STOCK PREDICTION METRICS");
        System.out.println("=".repeat(50));
        System.out.printf("MSE:   %.2f%n", mse);
        System.out.printf("RMSE:  %.2f%n", Math.sqrt(mse));
        System.out.printf("MAE:   %.2f%n", mae);
        System.out.printf("MAPE:  %.4f%%%n", mape);
        System.out.printf("R²:    %.4f%n", r2);
        System.out.println("=".repeat(50));

        double[] yTestData = yTest.getData();
        double[] predData = predictions.getData();

        System.out.println("\nGenerating prediction comparison charts...");
        
        ChartUtils.plotPredictions(yTestData, predData,
            "Stock Price Predictions vs Actual",
            outputDir + "/predictions.png",
            1200, 600
        );

        double[] errors = new double[yTestData.length];
        for (int i = 0; i < errors.length; i++) {
            errors[i] = yTestData[i] - predData[i];
        }
        
        ChartUtils.plotTimeSeries(errors, "Prediction Errors", outputDir + "/errors.png");
        ChartUtils.plotHistogram(errors, "Error Distribution", "Error", 
            outputDir + "/error_histogram.png", 25);

        double[] absErrors = new double[errors.length];
        for (int i = 0; i < errors.length; i++) {
            absErrors[i] = Math.abs(errors[i]);
        }
        ChartUtils.plotTimeSeries(absErrors, "Absolute Prediction Errors", 
            outputDir + "/abs_errors.png");

        System.out.println("\nGenerating trading strategy backtest...");
        backtestTradingStrategy(yTestData, predData, outputDir);

        System.out.println("\nSaving model...");
        com.tenzor.io.ModelSaver.save(model, outputDir + "/final_model.json");

        System.out.println("\n" + "=".repeat(80));
        System.out.println("STOCK PREDICTION COMPLETE");
        System.out.println("=".repeat(80));
        System.out.println("\nOutput files saved to: " + new File(outputDir).getAbsolutePath());
    }

    private static double[] generateSyntheticStockData(int length) {
        double[] data = new double[length];
        double price = 100.0;
        Random rand = new Random(42);
        
        for (int i = 0; i < length; i++) {
            double trend = 0.0005;
            double seasonality = 0.02 * Math.sin(i * 2 * Math.PI / 50);
            double noise = 0.015 * rand.nextGaussian();
            
            double return_ = trend + seasonality + noise;
            price *= (1 + return_);
            
            data[i] = Math.max(price, 1.0);
        }
        
        return data;
    }

    private static double[] generateVolume(int length) {
        double[] volume = new double[length];
        Random rand = new Random(123);
        for (int i = 0; i < length; i++) {
            volume[i] = 1000000 + 500000 * rand.nextGaussian();
        }
        return volume;
    }

    private static void backtestTradingStrategy(double[] actual, double[] predicted, String outputDir) {
        double initialCapital = 10000.0;
        double capital = initialCapital;
        double position = 0.0;
        double entryPrice = 0.0;
        
        List<Double> equityCurve = new ArrayList<>();
        List<Integer> trades = new ArrayList<>();
        int winTrades = 0;
        int totalTrades = 0;
        
        for (int i = 1; i < actual.length - 1; i++) {
            double predChange = (predicted[i] - predicted[i-1]) / predicted[i-1];
            double actualChange = (actual[i] - actual[i-1]) / actual[i-1];
            
            if (predChange > 0.01 && position == 0.0) {
                position = capital / actual[i];
                entryPrice = actual[i];
                capital = 0.0;
                totalTrades++;
            } else if (predChange < -0.01 && position > 0.0) {
                capital = position * actual[i];
                double profitPct = (actual[i] - entryPrice) / entryPrice;
                if (profitPct > 0) winTrades++;
                trades.add((int)(profitPct * 100));
                position = 0.0;
            }
            
            double currentEquity = capital + position * actual[i];
            equityCurve.add(currentEquity);
        }
        
        if (position > 0.0) {
            capital = position * actual[actual.length - 1];
            position = 0.0;
        }
        
        double finalReturn = (capital - initialCapital) / initialCapital * 100;
        double winRate = totalTrades > 0 ? (double) winTrades / totalTrades * 100 : 0;
        
        System.out.println("\n" + "=".repeat(50));
        System.out.println("TRADING STRATEGY BACKTEST");
        System.out.println("=".repeat(50));
        System.out.printf("Initial Capital:  $%.2f%n", initialCapital);
        System.out.printf("Final Capital:    $%.2f%n", capital);
        System.out.printf("Total Return:     %.2f%%%n", finalReturn);
        System.out.printf("Total Trades:     %d%n", totalTrades);
        System.out.printf("Winning Trades:   %d%n", winTrades);
        System.out.printf("Win Rate:         %.1f%%%n", winRate);
        System.out.println("=".repeat(50));
        
        double[] equityArr = new double[equityCurve.size()];
        for (int i = 0; i < equityArr.length; i++) {
            equityArr[i] = equityCurve.get(i);
        }
        ChartUtils.plotTimeSeries(equityArr, "Equity Curve", 
            outputDir + "/equity_curve.png");
    }
}
