package com.tenzor.data;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import java.util.*;

public class DataLoader {
    private Dataset dataset;
    private int batchSize;
    private boolean shuffle;
    private Random rng;

    public DataLoader(Dataset dataset, int batchSize) {
        this(dataset, batchSize, true);
    }

    public DataLoader(Dataset dataset, int batchSize, boolean shuffle) {
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.rng = new Random(42);
    }

    public List<Tensor[]> load() {
        int n = dataset.size();
        List<Tensor[]> batches = new ArrayList<>();

        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        if (shuffle) Collections.shuffle(Arrays.asList(indices), rng);

        for (int i = 0; i < n; i += batchSize) {
            int end = Math.min(i + batchSize, n);
            int batchLen = end - i;

            Tensor firstX = dataset.getX(indices[i]);
            Tensor firstY = dataset.getY(indices[i]);
            int[] xDims = firstX.shape().dims();
            int[] yDims = firstY.shape().dims();

            int[] xBatchDims = new int[xDims.length];
            xBatchDims[0] = batchLen;
            System.arraycopy(xDims, 1, xBatchDims, 1, xDims.length - 1);

            int[] yBatchDims = new int[yDims.length];
            yBatchDims[0] = batchLen;
            System.arraycopy(yDims, 1, yBatchDims, 1, yDims.length - 1);

            int xRowSize = firstX.size();
            int yRowSize = firstY.size();

            double[] xData = new double[batchLen * xRowSize];
            double[] yData = new double[batchLen * yRowSize];

            for (int j = 0; j < batchLen; j++) {
                Tensor xSample = dataset.getX(indices[i + j]);
                Tensor ySample = dataset.getY(indices[i + j]);
                System.arraycopy(xSample.getData(), 0, xData, j * xRowSize, xRowSize);
                System.arraycopy(ySample.getData(), 0, yData, j * yRowSize, yRowSize);
            }

            Tensor xBatch = new Tensor(xData, new Shape(xBatchDims));
            Tensor yBatch = new Tensor(yData, new Shape(yBatchDims));
            batches.add(new Tensor[]{xBatch, yBatch});
        }

        return batches;
    }

    public int getBatchSize() { return batchSize; }
}
