package com.tenzor.datasets;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import java.io.*;
import java.net.URL;
import java.nio.file.*;

public class CIFAR10 {
    private static final String URL_STR = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    private Tensor xTrain, yTrain, xTest, yTest;

    public CIFAR10() throws IOException {
        this(System.getProperty("java.io.tmpdir") + "/cifar10");
    }

    public CIFAR10(String dataDir) throws IOException {
        Path dir = Paths.get(dataDir);
        Files.createDirectories(dir);
        Path archive = dir.resolve("cifar-10-binary.tar.gz");

        if (!Files.exists(archive)) {
            System.out.println("Downloading CIFAR-10...");
            try (InputStream in = new URL(URL_STR).openStream()) {
                Files.copy(in, archive, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        double[][] xTrainList = new double[50000][3072];
        double[] yTrainList = new double[50000];
        double[][] xTestList = new double[10000][3072];
        double[] yTestList = new double[10000];

        for (int i = 1; i <= 5; i++) {
            Path batchFile = dir.resolve("data_batch_" + i + ".bin");
            if (!Files.exists(batchFile)) extractTarGz(archive, dir);
            try (DataInputStream dis = new DataInputStream(new FileInputStream(batchFile.toFile()))) {
                for (int j = 0; j < 10000; j++) {
                    yTrainList[(i - 1) * 10000 + j] = dis.readUnsignedByte();
                    for (int k = 0; k < 3072; k++) {
                        xTrainList[(i - 1) * 10000 + j][k] = dis.readUnsignedByte() / 255.0;
                    }
                }
            }
        }

        Path testBatch = dir.resolve("test_batch.bin");
        if (!Files.exists(testBatch)) extractTarGz(archive, dir);
        try (DataInputStream dis = new DataInputStream(new FileInputStream(testBatch.toFile()))) {
            for (int j = 0; j < 10000; j++) {
                yTestList[j] = dis.readUnsignedByte();
                for (int k = 0; k < 3072; k++) {
                    xTestList[j][k] = dis.readUnsignedByte() / 255.0;
                }
            }
        }

        xTrain = new Tensor(xTrainList);
        yTrain = new Tensor(yTrainList, new Shape(50000));
        xTest = new Tensor(xTestList);
        yTest = new Tensor(yTestList, new Shape(10000));
    }

    private void extractTarGz(Path archive, Path dest) {
        try {
            ProcessBuilder pb = new ProcessBuilder("tar", "xzf", archive.toString(), "-C", dest.toString());
            pb.inheritIO();
            Process p = pb.start();
            p.waitFor();
        } catch (Exception e) {
            System.err.println("Warning: Could not extract " + archive + ": " + e.getMessage());
        }
    }

    public Tensor getXTrain() { return xTrain; }
    public Tensor getYTrain() { return yTrain; }
    public Tensor getXTest() { return xTest; }
    public Tensor getYTest() { return yTest; }
}
