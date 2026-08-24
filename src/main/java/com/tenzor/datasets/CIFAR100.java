package com.tenzor.datasets;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import java.io.*;
import java.net.URL;
import java.nio.file.*;

public class CIFAR100 {
    private static final String URL_STR = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
    private Tensor xTrain, yTrain, xTest, yTest;

    public CIFAR100() throws IOException {
        this(System.getProperty("java.io.tmpdir") + "/cifar100");
    }

    public CIFAR100(String dataDir) throws IOException {
        Path dir = Paths.get(dataDir);
        Files.createDirectories(dir);
        Path archive = dir.resolve("cifar-100-binary.tar.gz");

        if (!Files.exists(archive)) {
            System.out.println("Downloading CIFAR-100...");
            try (InputStream in = new URL(URL_STR).openStream()) {
                Files.copy(in, archive, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        Path trainBin = dir.resolve("train.bin");
        Path testBin = dir.resolve("test.bin");
        if (!Files.exists(trainBin)) extractTarGz(archive, dir);

        try (DataInputStream dis = new DataInputStream(new FileInputStream(trainBin.toFile()))) {
            double[][] xData = new double[50000][3072];
            double[] yData = new double[50000];
            for (int i = 0; i < 50000; i++) {
                yData[i] = dis.readUnsignedByte();
                dis.readUnsignedByte();
                for (int j = 0; j < 3072; j++) {
                    xData[i][j] = dis.readUnsignedByte() / 255.0;
                }
            }
            xTrain = new Tensor(xData);
            yTrain = new Tensor(yData, new Shape(50000));
        }

        try (DataInputStream dis = new DataInputStream(new FileInputStream(testBin.toFile()))) {
            double[][] xData = new double[10000][3072];
            double[] yData = new double[10000];
            for (int i = 0; i < 10000; i++) {
                yData[i] = dis.readUnsignedByte();
                dis.readUnsignedByte();
                for (int j = 0; j < 3072; j++) {
                    xData[i][j] = dis.readUnsignedByte() / 255.0;
                }
            }
            xTest = new Tensor(xData);
            yTest = new Tensor(yData, new Shape(10000));
        }
    }

    private void extractTarGz(Path archive, Path dest) {
        try {
            ProcessBuilder pb = new ProcessBuilder("tar", "xzf", archive.toString(), "-C", dest.toString());
            pb.inheritIO();
            Process p = pb.start();
            p.waitFor();
        } catch (Exception e) {
            System.err.println("Warning: Could not extract " + archive);
        }
    }

    public Tensor getXTrain() { return xTrain; }
    public Tensor getYTrain() { return yTrain; }
    public Tensor getXTest() { return xTest; }
    public Tensor getYTest() { return yTest; }
}
