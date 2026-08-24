package com.tenzor.datasets;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import java.io.*;
import java.net.URL;
import java.nio.file.*;
import java.util.*;

public class Iris {
    private static final String DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
    private Tensor xTrain, yTrain, xTest, yTest;
    private static final Map<String, Integer> LABEL_MAP = new HashMap<>();
    static {
        LABEL_MAP.put("Iris-setosa", 0);
        LABEL_MAP.put("Iris-versicolor", 1);
        LABEL_MAP.put("Iris-virginica", 2);
    }

    public Iris() throws IOException {
        this(System.getProperty("java.io.tmpdir") + "/iris", 0.8);
    }

    public Iris(String dataDir, double trainRatio) throws IOException {
        Path dir = Paths.get(dataDir);
        Files.createDirectories(dir);
        Path dataFile = dir.resolve("iris.data");

        if (!Files.exists(dataFile)) {
            System.out.println("Downloading Iris...");
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
                String[] parts = line.split(",");
                double[] features = new double[4];
                for (int i = 0; i < 4; i++) features[i] = Double.parseDouble(parts[i]);
                allX.add(features);
                Integer label = LABEL_MAP.get(parts[4]);
                allY.add(label != null ? (double) label : -1);
            }
        }

        int n = allX.size();
        int trainSize = (int) (n * trainRatio);

        double[][] xTrainData = new double[trainSize][4];
        double[] yTrainData = new double[trainSize];
        double[][] xTestData = new double[n - trainSize][4];
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
