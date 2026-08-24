package com.tenzor.data;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;

public class TensorDataset implements Dataset {
    private Tensor x;
    private Tensor y;

    public TensorDataset(Tensor x, Tensor y) {
        if (x.shape().dim(0) != y.shape().dim(0)) {
            throw new IllegalArgumentException("X and Y must have same number of samples");
        }
        this.x = x;
        this.y = y;
    }

    @Override
    public int size() { return x.shape().dim(0); }

    @Override
    public Tensor getX(int index) {
        return x.slice(0, index, index + 1);
    }

    @Override
    public Tensor getY(int index) {
        if (y.shape().numDims() == 1) {
            return new Tensor(y.get(index));
        }
        return y.slice(0, index, index + 1);
    }

    public Tensor getX() { return x; }
    public Tensor getY() { return y; }
}
