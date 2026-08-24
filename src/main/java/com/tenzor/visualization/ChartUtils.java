package com.tenzor.visualization;

import org.jfree.chart.*;
import org.jfree.chart.axis.*;
import org.jfree.chart.plot.*;
import org.jfree.chart.renderer.xy.*;
import org.jfree.data.xy.*;
import org.jfree.data.category.*;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;

public class ChartUtils {

    public static void plotTrainingHistory(Map<String, List<Double>> history, String title, String outputPath) {
        plotTrainingHistory(history, title, outputPath, 800, 600);
    }

    public static void plotTrainingHistory(Map<String, List<Double>> history, String title, 
                                            String outputPath, int width, int height) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        
        if (history.containsKey("loss")) {
            XYSeries trainLoss = new XYSeries("Train Loss");
            List<Double> loss = history.get("loss");
            for (int i = 0; i < loss.size(); i++) {
                trainLoss.add(i + 1, loss.get(i));
            }
            dataset.addSeries(trainLoss);
        }
        
        if (history.containsKey("val_loss")) {
            XYSeries valLoss = new XYSeries("Validation Loss");
            List<Double> loss = history.get("val_loss");
            for (int i = 0; i < loss.size(); i++) {
                valLoss.add(i + 1, loss.get(i));
            }
            dataset.addSeries(valLoss);
        }
        
        if (history.containsKey("accuracy")) {
            XYSeries acc = new XYSeries("Train Accuracy");
            List<Double> accList = history.get("accuracy");
            for (int i = 0; i < accList.size(); i++) {
                acc.add(i + 1, accList.get(i));
            }
            dataset.addSeries(acc);
        }
        
        if (history.containsKey("val_accuracy")) {
            XYSeries valAcc = new XYSeries("Validation Accuracy");
            List<Double> accList = history.get("val_accuracy");
            for (int i = 0; i < accList.size(); i++) {
                valAcc.add(i + 1, accList.get(i));
            }
            dataset.addSeries(valAcc);
        }

        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            "Epoch",
            "Value",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        for (int i = 0; i < dataset.getSeriesCount(); i++) {
            renderer.setSeriesLinesVisible(i, true);
            renderer.setSeriesShapesVisible(i, true);
        }
        plot.setRenderer(renderer);

        try {
            saveChart(chart, outputPath, width, height);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }

    public static void plotPredictions(double[] actual, double[] predicted, String title, String outputPath) {
        plotPredictions(actual, predicted, title, outputPath, 800, 600);
    }

    public static void plotPredictions(double[] actual, double[] predicted, String title, 
                                        String outputPath, int width, int height) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        
        XYSeries actualSeries = new XYSeries("Actual");
        XYSeries predictedSeries = new XYSeries("Predicted");
        
        int n = Math.min(actual.length, predicted.length);
        for (int i = 0; i < n; i++) {
            actualSeries.add(i, actual[i]);
            predictedSeries.add(i, predicted[i]);
        }
        
        dataset.addSeries(actualSeries);
        dataset.addSeries(predictedSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            "Sample",
            "Value",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        try {
            saveChart(chart, outputPath, width, height);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }

    public static void plotTimeSeries(double[] data, String title, String outputPath) {
        plotTimeSeries(data, title, outputPath, 800, 600);
    }

    public static void plotTimeSeries(double[] data, String title, String outputPath, 
                                        int width, int height) {
        XYSeries series = new XYSeries("Data");
        for (int i = 0; i < data.length; i++) {
            series.add(i, data[i]);
        }
        
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);

        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            "Time Step",
            "Value",
            dataset,
            PlotOrientation.VERTICAL,
            false,
            true,
            false
        );

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        try {
            saveChart(chart, outputPath, width, height);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }

    public static void plotBarChart(String[] categories, double[] values, String title, 
                                     String categoryLabel, String valueLabel, String outputPath) {
        plotBarChart(categories, values, title, categoryLabel, valueLabel, outputPath, 800, 600);
    }

    public static void plotBarChart(String[] categories, double[] values, String title, 
                                     String categoryLabel, String valueLabel, String outputPath,
                                     int width, int height) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < categories.length && i < values.length; i++) {
            dataset.addValue(values[i], "Value", categories[i]);
        }

        JFreeChart chart = ChartFactory.createBarChart(
            title,
            categoryLabel,
            valueLabel,
            dataset,
            PlotOrientation.VERTICAL,
            false,
            true,
            false
        );

        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        try {
            saveChart(chart, outputPath, width, height);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }

    public static void plotConfusionMatrix(int[][] matrix, String[] labels, String title, String outputPath) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < matrix.length && i < labels.length; i++) {
            for (int j = 0; j < matrix[i].length && j < labels.length; j++) {
                dataset.addValue(matrix[i][j], labels[i], labels[j]);
            }
        }

        JFreeChart chart = ChartFactory.createBarChart(
            title,
            "Predicted",
            "Actual",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        try {
            saveChart(chart, outputPath, 800, 600);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }

    public static void plotMultipleMetrics(List<double[]> seriesData, List<String> seriesNames,
                                            String title, String xLabel, String yLabel, String outputPath) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        
        for (int s = 0; s < seriesData.size() && s < seriesNames.size(); s++) {
            XYSeries series = new XYSeries(seriesNames.get(s));
            double[] data = seriesData.get(s);
            for (int i = 0; i < data.length; i++) {
                series.add(i, data[i]);
            }
            dataset.addSeries(series);
        }

        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            xLabel,
            yLabel,
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        try {
            saveChart(chart, outputPath, 800, 600);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }

    public static void plotScatter(double[] x, double[] y, String title, 
                                    String xLabel, String yLabel, String outputPath) {
        XYSeries series = new XYSeries("Data Points");
        int n = Math.min(x.length, y.length);
        for (int i = 0; i < n; i++) {
            series.add(x[i], y[i]);
        }
        
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);

        JFreeChart chart = ChartFactory.createScatterPlot(
            title,
            xLabel,
            yLabel,
            dataset,
            PlotOrientation.VERTICAL,
            false,
            true,
            false
        );

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        try {
            saveChart(chart, outputPath, 800, 600);
        } catch (IOException e) {
            System.err.println("Error saving chart: " + e.getMessage());
        }
    }

    public static void plotResiduals(double[] actual, double[] predicted, String title, String outputPath) {
        double[] residuals = new double[actual.length];
        double[] xIndices = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            residuals[i] = actual[i] - predicted[i];
            xIndices[i] = i;
        }
        plotScatter(xIndices, residuals, title, "Sample Index", "Residual", outputPath);
    }

    public static void plotHistogram(double[] data, String title, String xLabel, 
                                      String outputPath, int bins) {
        if (data.length == 0) return;
        
        double min = data[0];
        double max = data[0];
        for (double v : data) {
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        
        double binWidth = (max - min) / bins;
        int[] counts = new int[bins];
        
        for (double v : data) {
            int bin = (int) ((v - min) / binWidth);
            if (bin >= bins) bin = bins - 1;
            if (bin < 0) bin = 0;
            counts[bin]++;
        }
        
        String[] binLabels = new String[bins];
        double[] countsDouble = new double[bins];
        for (int i = 0; i < bins; i++) {
            binLabels[i] = String.format("%.2f-%.2f", min + i * binWidth, min + (i + 1) * binWidth);
            countsDouble[i] = counts[i];
        }
        
        plotBarChart(binLabels, countsDouble, title, xLabel, "Frequency", outputPath);
    }

    private static void saveChart(JFreeChart chart, String outputPath, int width, int height) throws IOException {
        File file = new File(outputPath);
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        
        if (outputPath.toLowerCase().endsWith(".png")) {
            org.jfree.chart.ChartUtils.saveChartAsPNG(file, chart, width, height);
        } else if (outputPath.toLowerCase().endsWith(".jpg") || outputPath.toLowerCase().endsWith(".jpeg")) {
            org.jfree.chart.ChartUtils.saveChartAsJPEG(file, chart, width, height);
        } else {
            org.jfree.chart.ChartUtils.saveChartAsPNG(file, chart, width, height);
        }
        System.out.println("Chart saved to: " + outputPath);
    }
}
