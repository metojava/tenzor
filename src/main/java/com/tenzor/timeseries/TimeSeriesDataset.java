package com.tenzor.timeseries;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import java.util.*;

public class TimeSeriesDataset {
    private double[] data;
    private int sequenceLength;
    private int forecastHorizon;

    public TimeSeriesDataset(double[] data, int sequenceLength, int forecastHorizon) {
        this.data = data;
        this.sequenceLength = sequenceLength;
        this.forecastHorizon = forecastHorizon;
    }

    public Tensor[] generateSamples() {
        int n = data.length - sequenceLength - forecastHorizon + 1;
        double[][] xData = new double[n][sequenceLength];
        double[][] yData = new double[n][forecastHorizon];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                xData[i][j] = data[i + j];
            }
            for (int j = 0; j < forecastHorizon; j++) {
                yData[i][j] = data[i + sequenceLength + j];
            }
        }

        Tensor x = new Tensor(xData);
        Tensor y = forecastHorizon == 1 ?
            new Tensor(Arrays.stream(yData).mapToDouble(d -> d[0]).toArray(), new Shape(n)) :
            new Tensor(yData);

        return new Tensor[]{x, y};
    }

    public Tensor[] trainTestSplit(double trainRatio) {
        Tensor[] samples = generateSamples();
        Tensor x = samples[0], y = samples[1];
        int trainSize = (int) (x.shape().dim(0) * trainRatio);

        Tensor xTrain = x.slice(0, 0, trainSize);
        Tensor yTrain = y.slice(0, 0, trainSize);
        Tensor xTest = x.slice(0, trainSize, x.shape().dim(0));
        Tensor yTest = y.slice(0, trainSize, y.shape().dim(0));

        return new Tensor[]{xTrain, yTrain, xTest, yTest};
    }
}
