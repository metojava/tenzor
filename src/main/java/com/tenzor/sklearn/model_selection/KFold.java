package com.tenzor.sklearn.model_selection;

import java.util.*;

public class KFold {
    private int nSplits;
    private boolean shuffle;
    private long randomSeed;

    public KFold() { this(5, true, 42); }

    public KFold(int nSplits, boolean shuffle, long randomSeed) {
        this.nSplits = nSplits;
        this.shuffle = shuffle;
        this.randomSeed = randomSeed;
    }

    public List<int[][]> split(int n) {
        List<int[][]> folds = new ArrayList<>();
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        if (shuffle) {
            Random rng = new Random(randomSeed);
            for (int i = n - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }

        int foldSize = n / nSplits;
        for (int fold = 0; fold < nSplits; fold++) {
            int start = fold * foldSize;
            int end = (fold == nSplits - 1) ? n : start + foldSize;

            int[] testIdx = Arrays.copyOfRange(indices, start, end);
            int[] trainIdx = new int[n - (end - start)];
            int idx = 0;
            for (int i = 0; i < start; i++) trainIdx[idx++] = indices[i];
            for (int i = end; i < n; i++) trainIdx[idx++] = indices[i];

            folds.add(new int[][]{trainIdx, testIdx});
        }

        return folds;
    }

    public int getNSplits() { return nSplits; }
}
