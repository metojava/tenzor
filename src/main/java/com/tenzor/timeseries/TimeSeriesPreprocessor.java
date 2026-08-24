package com.tenzor.timeseries;

import com.tenzor.core.Tensor;

public class TimeSeriesPreprocessor {
    private double mean;
    private double std;
    private double min;
    private double max;

    public TimeSeriesPreprocessor() {}

    public double[] normalizeZScore(double[] data) {
        mean = 0;
        for (double d : data) mean += d;
        mean /= data.length;
        std = 0;
        for (double d : data) std += (d - mean) * (d - mean);
        std = Math.sqrt(std / data.length);
        if (std < 1e-10) std = 1.0;

        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) result[i] = (data[i] - mean) / std;
        return result;
    }

    public double[] denormalizeZScore(double[] normalized) {
        double[] result = new double[normalized.length];
        for (int i = 0; i < normalized.length; i++) result[i] = normalized[i] * std + mean;
        return result;
    }

    public double[] normalizeMinMax(double[] data) {
        min = Double.MAX_VALUE;
        max = -Double.MAX_VALUE;
        for (double d : data) {
            if (d < min) min = d;
            if (d > max) max = d;
        }
        double range = max - min;
        if (range < 1e-10) range = 1.0;

        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) result[i] = (data[i] - min) / range;
        return result;
    }

    public double[] denormalizeMinMax(double[] normalized) {
        double range = max - min;
        double[] result = new double[normalized.length];
        for (int i = 0; i < normalized.length; i++) result[i] = normalized[i] * range + min;
        return result;
    }

    public double[] difference(double[] data, int order) {
        double[] current = data.clone();
        for (int o = 0; o < order; o++) {
            double[] diff = new double[current.length - 1];
            for (int i = 0; i < diff.length; i++) diff[i] = current[i + 1] - current[i];
            current = diff;
        }
        return current;
    }

    public double[] logTransform(double[] data) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = Math.log(Math.max(data[i], 1e-10));
        }
        return result;
    }

    public double[] expTransform(double[] data) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = Math.exp(data[i]);
        }
        return result;
    }

    public double getMean() { return mean; }
    public double getStd() { return std; }
}
