package com.tenzor.data;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class ImageDataset implements Dataset {
    private List<Tensor> images;
    private List<Double> labels;
    private int height;
    private int width;
    private int channels;

    public ImageDataset(String directory, boolean normalize) throws IOException {
        this.images = new ArrayList<>();
        this.labels = new ArrayList<>();
        File dir = new File(directory);
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IOException("Directory not found: " + directory);
        }
        File[] classDirs = dir.listFiles(File::isDirectory);
        if (classDirs != null) {
            int label = 0;
            for (File classDir : classDirs) {
                File[] files = classDir.listFiles((d, name) -> {
                    String lower = name.toLowerCase();
                    return lower.endsWith(".jpg") || lower.endsWith(".jpeg") ||
                           lower.endsWith(".png") || lower.endsWith(".bmp");
                });
                if (files != null) {
                    for (File file : files) {
                        BufferedImage img = ImageIO.read(file);
                        if (img != null) {
                            height = img.getHeight();
                            width = img.getWidth();
                            channels = img.getColorModel().getNumComponents();
                            double[] pixels = new double[height * width * channels];
                            int idx = 0;
                            for (int y = 0; y < height; y++) {
                                for (int x = 0; x < width; x++) {
                                    int rgb = img.getRGB(x, y);
                                    if (channels >= 3) {
                                        pixels[idx++] = normalize ? ((rgb >> 16) & 0xFF) / 255.0 : (rgb >> 16) & 0xFF;
                                        pixels[idx++] = normalize ? ((rgb >> 8) & 0xFF) / 255.0 : (rgb >> 8) & 0xFF;
                                        pixels[idx++] = (rgb & 0xFF) / (normalize ? 255.0 : 1.0);
                                    } else {
                                        int gray = (int) (0.299 * ((rgb >> 16) & 0xFF) + 0.587 * ((rgb >> 8) & 0xFF) + 0.114 * (rgb & 0xFF));
                                        pixels[idx++] = normalize ? gray / 255.0 : gray;
                                    }
                                }
                            }
                            images.add(new Tensor(pixels, new Shape(1, height, width, channels)));
                            labels.add((double) label);
                        }
                    }
                }
                label++;
            }
        }
    }

    @Override
    public int size() { return images.size(); }

    @Override
    public Tensor getX(int index) { return images.get(index); }

    @Override
    public Tensor getY(int index) { return new Tensor(labels.get(index)); }

    public int getHeight() { return height; }
    public int getWidth() { return width; }
    public int getChannels() { return channels; }
}
