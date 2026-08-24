package com.tenzor.data;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import java.io.*;
import java.util.*;

public class TextDataset implements Dataset {
    private List<String> texts;
    private List<Double> labels;
    private Map<String, Integer> vocabulary;
    private int maxLength;

    public TextDataset(String filePath, int maxLength) throws IOException {
        this.texts = new ArrayList<>();
        this.labels = new ArrayList<>();
        this.vocabulary = new HashMap<>();
        this.maxLength = maxLength;
        loadFile(filePath);
    }

    private void loadFile(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\t", 2);
                if (parts.length == 2) {
                    labels.add(Double.parseDouble(parts[0].trim()));
                    texts.add(parts[1].trim());
                }
            }
        }
        buildVocabulary();
    }

    private void buildVocabulary() {
        int idx = 1;
        for (String text : texts) {
            for (String word : text.toLowerCase().split("\\W+")) {
                if (!word.isEmpty() && !vocabulary.containsKey(word)) {
                    vocabulary.put(word, idx++);
                }
            }
        }
    }

    public Tensor textToTensor(String text) {
        double[] data = new double[maxLength];
        String[] words = text.toLowerCase().split("\\W+");
        for (int i = 0; i < maxLength && i < words.length; i++) {
            Integer idx = vocabulary.get(words[i]);
            data[i] = idx != null ? idx : 0;
        }
        return new Tensor(data, new Shape(1, maxLength));
    }

    @Override
    public int size() { return texts.size(); }

    @Override
    public Tensor getX(int index) {
        return textToTensor(texts.get(index));
    }

    @Override
    public Tensor getY(int index) {
        return new Tensor(labels.get(index));
    }

    public int vocabSize() { return vocabulary.size() + 1; }
    public int getMaxLength() { return maxLength; }
}
