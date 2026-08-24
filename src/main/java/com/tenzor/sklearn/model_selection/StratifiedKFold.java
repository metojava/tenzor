package com.tenzor.sklearn.model_selection;

import com.tenzor.core.Tensor;
import java.util.*;

public class StratifiedKFold {
    private int nSplits;
    private boolean shuffle;
    private long randomSeed;

    public StratifiedKFold() { this(5, true, 42); }

    public StratifiedKFold(int nSplits, boolean shuffle, long randomSeed) {
        this.nSplits = nSplits;
        this.shuffle = shuffle;
        this.randomSeed = randomSeed;
    }

    public List<int[][]> split(Tensor X, Tensor y) {
        int n = X.shape().dim(0);
        Map<Double, List<Integer>> classIndices = new HashMap<>();
        for (int i = 0; i < n; i++) {
            double label = y.get(i);
            classIndices.computeIfAbsent(label, k -> new ArrayList<>()).add(i);
        }

        List<int[][]> folds = new ArrayList<>();
        for (int i = 0; i < nSplits; i++) {
            folds.add(new int[][]{new int[0], new int[0]});
        }

        Random rng = new Random(randomSeed);
        for (Map.Entry<Double, List<Integer>> entry : classIndices.entrySet()) {
            List<Integer> indices = entry.getValue();
            if (shuffle) Collections.shuffle(indices, rng);

            int perFold = indices.size() / nSplits;
            int remainder = indices.size() % nSplits;

            int foldIdx = 0;
            int pos = 0;
            for (int f = 0; f < nSplits; f++) {
                int count = perFold + (f < remainder ? 1 : 0);
                int[] testPart = new int[count];
                for (int j = 0; j < count; j++) {
                    testPart[j] = indices.get(pos++);
                }
                int[][] fold = folds.get(f);
                int[] existingTest = fold[1];
                int[] newTest = Arrays.copyOf(existingTest, existingTest.length + count);
                System.arraycopy(testPart, 0, newTest, existingTest.length, count);
                folds.set(f, new int[][]{fold[0], newTest});
            }
        }

        List<int[][]> result = new ArrayList<>();
        for (int f = 0; f < nSplits; f++) {
            int[][] fold = folds.get(f);
            int[] testIdx = fold[1];
            Set<Integer> testSet = new HashSet<>();
            for (int idx : testIdx) testSet.add(idx);

            int[] trainIdx = new int[n - testIdx.length];
            int idx = 0;
            for (int i = 0; i < n; i++) {
                if (!testSet.contains(i)) trainIdx[idx++] = i;
            }
            result.add(new int[][]{trainIdx, testIdx});
        }

        return result;
    }

    public int getNSplits() { return nSplits; }
}
