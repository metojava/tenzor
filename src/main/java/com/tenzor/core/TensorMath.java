package com.tenzor.core;

import java.util.Random;

public final class TensorMath {

    private TensorMath() {}

    public static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double leakyRelu(double x, double alpha) {
        return x >= 0 ? x : alpha * x;
    }

    public static double elu(double x, double alpha) {
        return x >= 0 ? x : alpha * (Math.exp(x) - 1);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static Tensor xavierUniform(int... shape) {
        Shape s = new Shape(shape);
        int fanIn = s.dim(s.numDims() - 2);
        int fanOut = s.dim(s.numDims() - 1);
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        Random rng = new Random();
        double[] data = new double[s.totalSize()];
        for (int i = 0; i < data.length; i++) {
            data[i] = -limit + rng.nextDouble() * 2 * limit;
        }
        return new Tensor(data, s);
    }

    public static Tensor xavierNormal(int... shape) {
        Shape s = new Shape(shape);
        int fanIn = s.dim(s.numDims() - 2);
        int fanOut = s.dim(s.numDims() - 1);
        double std = Math.sqrt(2.0 / (fanIn + fanOut));
        return Tensor.randomNormal(s, 0, std);
    }

    public static Tensor heUniform(int... shape) {
        Shape s = new Shape(shape);
        int fanIn = s.dim(s.numDims() - 2);
        double limit = Math.sqrt(6.0 / fanIn);
        Random rng = new Random();
        double[] data = new double[s.totalSize()];
        for (int i = 0; i < data.length; i++) {
            data[i] = -limit + rng.nextDouble() * 2 * limit;
        }
        return new Tensor(data, s);
    }

    public static Tensor heNormal(int... shape) {
        Shape s = new Shape(shape);
        int fanIn = s.dim(s.numDims() - 2);
        double std = Math.sqrt(2.0 / fanIn);
        return Tensor.randomNormal(s, 0, std);
    }

    public static Tensor dot(Tensor a, Tensor b) {
        return a.matmul(b);
    }

    public static Tensor concatenate(Tensor a, Tensor b, int axis) {
        int[] aDims = a.shape().dims();
        int[] bDims = b.shape().dims();
        if (aDims.length != bDims.length) {
            throw new IllegalArgumentException("Tensors must have same number of dimensions");
        }
        if (axis < 0) axis += aDims.length;
        for (int i = 0; i < aDims.length; i++) {
            if (i != axis && aDims[i] != bDims[i]) {
                throw new IllegalArgumentException("Tensors must match on all dimensions except concatenation axis");
            }
        }
        int[] newDims = aDims.clone();
        newDims[axis] = aDims[axis] + bDims[axis];
        Shape newShape = new Shape(newDims);
        double[] result = new double[newShape.totalSize()];
        int strideBefore = 1, strideAfter = 1, aSize = aDims[axis], bSize = bDims[axis];
        for (int i = 0; i < axis; i++) strideBefore *= aDims[i];
        for (int i = axis + 1; i < aDims.length; i++) strideAfter *= aDims[i];
        int blockA = aSize * strideAfter;
        int blockB = bSize * strideAfter;
        int totalBlock = (aSize + bSize) * strideAfter;
        for (int bIdx = 0; bIdx < strideBefore; bIdx++) {
            int base = bIdx * totalBlock;
            System.arraycopy(a.data, bIdx * blockA, result, base, blockA);
            System.arraycopy(b.data, bIdx * blockB, result, base + blockA, blockB);
        }
        return new Tensor(result, newShape);
    }

    public static Tensor pad(Tensor input, int[] padBefore, int[] padAfter, double padValue) {
        int[] dims = input.shape().dims();
        int[] newDims = new int[dims.length];
        for (int i = 0; i < dims.length; i++) {
            newDims[i] = dims[i] + padBefore[i] + padAfter[i];
        }
        Shape newShape = new Shape(newDims);
        double[] result = new double[newShape.totalSize()];
        Arrays.fill(result, padValue);
        int[] srcStrides = new int[dims.length];
        srcStrides[dims.length - 1] = 1;
        for (int i = dims.length - 2; i >= 0; i--) srcStrides[i] = srcStrides[i + 1] * dims[i + 1];
        int[] dstStrides = new int[newDims.length];
        dstStrides[newDims.length - 1] = 1;
        for (int i = newDims.length - 2; i >= 0; i--) dstStrides[i] = dstStrides[i + 1] * newDims[i + 1];
        for (int i = 0; i < input.data.length; i++) {
            int[] indices = input.shape().indicesOf(i);
            int dstIdx = 0;
            for (int d = 0; d < indices.length; d++) {
                dstIdx += (indices[d] + padBefore[d]) * dstStrides[d];
            }
            result[dstIdx] = input.data[i];
        }
        return new Tensor(result, newShape);
    }

    private static class Arrays {
        static void fill(double[] a, double val) {
            for (int i = 0; i < a.length; i++) a[i] = val;
        }
    }
}
