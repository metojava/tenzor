package com.tenzor.timeseries;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;

public class SequenceGenerator {

    public static Tensor[] generateSequences(double[] data, int seqLength, int horizon) {
        int n = data.length - seqLength - horizon + 1;
        if (n <= 0) throw new IllegalArgumentException("Data too short for sequence length and horizon");

        double[][] xData = new double[n][seqLength];
        double[][] yData = new double[n][horizon];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < seqLength; j++) {
                xData[i][j] = data[i + j];
            }
            for (int j = 0; j < horizon; j++) {
                yData[i][j] = data[i + seqLength + j];
            }
        }

        Tensor x = new Tensor(xData);
        Tensor y = horizon == 1 ?
            new Tensor(java.util.Arrays.stream(yData).mapToDouble(d -> d[0]).toArray(), new Shape(n)) :
            new Tensor(yData);

        return new Tensor[]{x, y};
    }

    public static Tensor[] generateSlidingWindow(double[] data, int windowSize) {
        return generateSequences(data, windowSize, 1);
    }

    public static double[] generateSineWave(int length, double freq, double amplitude, double phase) {
        double[] data = new double[length];
        for (int i = 0; i < length; i++) {
            data[i] = amplitude * Math.sin(2 * Math.PI * freq * i / length + phase);
        }
        return data;
    }

    public static double[] generateRandomWalk(int length, double mean, double std) {
        java.util.Random rng = new java.util.Random(42);
        double[] data = new double[length];
        data[0] = mean;
        for (int i = 1; i < length; i++) {
            data[i] = data[i - 1] + rng.nextGaussian() * std;
        }
        return data;
    }
}
