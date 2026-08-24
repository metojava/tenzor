package com.tenzor.datasets;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import java.io.*;
import java.net.URL;
import java.nio.file.*;
import java.util.zip.GZIPInputStream;

public class FashionMNIST {
    private static final String BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/";
    private static final String[] FILES = {
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    };

    private Tensor xTrain, yTrain, xTest, yTest;

    public FashionMNIST() throws IOException {
        this(System.getProperty("java.io.tmpdir") + "/fashion_mnist");
    }

    public FashionMNIST(String dataDir) throws IOException {
        Path dir = Paths.get(dataDir);
        Files.createDirectories(dir);

        for (String file : FILES) {
            Path filePath = dir.resolve(file);
            if (!Files.exists(filePath)) {
                System.out.println("Downloading " + file + "...");
                try (InputStream in = new URL(BASE_URL + file).openStream()) {
                    Files.copy(in, filePath, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }

        xTrain = loadImages(dir.resolve(FILES[0]).toString());
        yTrain = loadLabels(dir.resolve(FILES[1]).toString());
        xTest = loadImages(dir.resolve(FILES[2]).toString());
        yTest = loadLabels(dir.resolve(FILES[3]).toString());
    }

    private Tensor loadImages(String gzPath) throws IOException {
        try (InputStream gzip = new GZIPInputStream(new FileInputStream(gzPath));
             DataInputStream dis = new DataInputStream(gzip)) {
            int magic = dis.readInt();
            int numImages = dis.readInt();
            int rows = dis.readInt();
            int cols = dis.readInt();

            double[] data = new double[numImages * rows * cols];
            for (int i = 0; i < data.length; i++) {
                data[i] = (dis.readUnsignedByte()) / 255.0;
            }
            return new Tensor(data, new Shape(numImages, rows * cols));
        }
    }

    private Tensor loadLabels(String gzPath) throws IOException {
        try (InputStream gzip = new GZIPInputStream(new FileInputStream(gzPath));
             DataInputStream dis = new DataInputStream(gzip)) {
            int magic = dis.readInt();
            int numLabels = dis.readInt();

            double[] data = new double[numLabels];
            for (int i = 0; i < numLabels; i++) {
                data[i] = dis.readUnsignedByte();
            }
            return new Tensor(data, new Shape(numLabels));
        }
    }

    public Tensor getXTrain() { return xTrain; }
    public Tensor getYTrain() { return yTrain; }
    public Tensor getXTest() { return xTest; }
    public Tensor getYTest() { return yTest; }
}
