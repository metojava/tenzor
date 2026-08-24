package com.tenzor.datasets;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import java.io.*;
import java.net.URL;
import java.nio.file.*;
import java.util.*;

public class BostonHousing {
    private static final String DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data";
    private Tensor xTrain, yTrain, xTest, yTest;

    public BostonHousing() throws IOException {
        this(System.getProperty("java.io.tmpdir") + "/boston_housing", 0.8);
    }

    public BostonHousing(String dataDir, double trainRatio) throws IOException {
        Path dir = Paths.get(dataDir);
        Files.createDirectories(dir);
        Path dataFile = dir.resolve("housing.data");

        if (!Files.exists(dataFile)) {
            System.out.println("Downloading Boston Housing...");
            try (InputStream in = new URL(DATA_URL).openStream()) {
                Files.copy(in, dataFile, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        List<double[]> allX = new ArrayList<>();
        List<Double> allY = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(dataFile.toFile()))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("\\s+");
                double[] features = new double[parts.length - 1];
                for (int i = 0; i < parts.length - 1; i++) {
                    features[i] = Double.parseDouble(parts[i]);
                }
                allX.add(features);
                allY.add(Double.parseDouble(parts[parts.length - 1]));
            }
        }

        int n = allX.size();
        int m = allX.get(0).length;
        int trainSize = (int) (n * trainRatio);

        double[][] xTrainData = new double[trainSize][m];
        double[] yTrainData = new double[trainSize];
        double[][] xTestData = new double[n - trainSize][m];
        double[] yTestData = new double[n - trainSize];

        for (int i = 0; i < trainSize; i++) {
            xTrainData[i] = allX.get(i);
            yTrainData[i] = allY.get(i);
        }
        for (int i = trainSize; i < n; i++) {
            xTestData[i - trainSize] = allX.get(i);
            yTestData[i - trainSize] = allY.get(i);
        }

        xTrain = new Tensor(xTrainData);
        yTrain = new Tensor(yTrainData, new Shape(trainSize));
        xTest = new Tensor(xTestData);
        yTest = new Tensor(yTestData, new Shape(n - trainSize));
    }

    public Tensor getXTrain() { return xTrain; }
    public Tensor getYTrain() { return yTrain; }
    public Tensor getXTest() { return xTest; }
    public Tensor getYTest() { return yTest; }
}
