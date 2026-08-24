package com.tenzor.core;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

public class Tensor {
    protected Shape shape;
    protected double[] data;

    public Tensor() {
        this(new double[]{0}, new Shape(1));
    }

    public Tensor(double value) {
        this(new double[]{value}, new Shape(1));
    }

    public Tensor(double[] data) {
        this(data, data.length > 0 ? new Shape(data.length) : new Shape(1));
        if (data.length == 0) this.data = new double[]{0};
    }

    public Tensor(double[][] data) {
        int rows = data.length;
        int cols = rows > 0 ? data[0].length : 0;
        this.shape = new Shape(rows, cols);
        this.data = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            if (data[i].length != cols) throw new IllegalArgumentException("Irregular 2D array");
            System.arraycopy(data[i], 0, this.data, i * cols, cols);
        }
    }

    public Tensor(double[][][] data) {
        int d1 = data.length;
        int d2 = d1 > 0 ? data[0].length : 0;
        int d3 = d2 > 0 ? data[0][0].length : 0;
        this.shape = new Shape(d1, d2, d3);
        this.data = new double[d1 * d2 * d3];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            if (data[i].length != d2) throw new IllegalArgumentException("Irregular 3D array");
            for (int j = 0; j < d2; j++) {
                if (data[i][j].length != d3) throw new IllegalArgumentException("Irregular 3D array");
                System.arraycopy(data[i][j], 0, this.data, idx, d3);
                idx += d3;
            }
        }
    }

    public Tensor(double[] data, Shape shape) {
        if (data.length != shape.totalSize()) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " does not match shape " + shape);
        }
        this.data = data.clone();
        this.shape = shape;
    }

    public Tensor(Tensor other) {
        this.shape = other.shape;
        this.data = other.data.clone();
    }

    public Shape shape() { return shape; }

    public int[] dims() { return shape.dims(); }

    public int numDims() { return shape.numDims(); }

    public int size() { return shape.totalSize(); }

    public double[] getData() { return data; }

    public double scalar() {
        if (shape.totalSize() != 1) throw new IllegalStateException("Not a scalar tensor");
        return data[0];
    }

    public double get(int... indices) {
        return data[shape.index(indices)];
    }

    public void set(double value, int... indices) {
        data[shape.index(indices)] = value;
    }

    public Tensor slice(int dim, int start, int end) {
        int[] newDims = shape.dims();
        newDims[dim] = end - start;
        Shape newShape = new Shape(newDims);
        double[] newData = new double[newShape.totalSize()];
        int stride = 1;
        for (int i = shape.numDims() - 1; i > dim; i--) stride *= shape.dim(i);
        int blockSize = stride * shape.dim(dim);
        int newBlockSize = stride * (end - start);
        int srcIdx = start * stride;
        int destIdx = 0;
        int totalBlocks = 1;
        for (int i = 0; i < dim; i++) totalBlocks *= shape.dim(i);
        for (int b = 0; b < totalBlocks; b++) {
            int base = b * blockSize;
            System.arraycopy(data, base + srcIdx, newData, destIdx, newBlockSize);
            destIdx += newBlockSize;
        }
        return new Tensor(newData, newShape);
    }

    public static Tensor zeros(Shape shape) {
        return new Tensor(new double[shape.totalSize()], shape);
    }

    public static Tensor zeros(int... shape) {
        return zeros(new Shape(shape));
    }

    public static Tensor ones(Shape shape) {
        double[] data = new double[shape.totalSize()];
        Arrays.fill(data, 1.0);
        return new Tensor(data, shape);
    }

    public static Tensor ones(int... shape) {
        return ones(new Shape(shape));
    }

    public static Tensor random(Shape shape) {
        Random rng = new Random();
        double[] data = new double[shape.totalSize()];
        for (int i = 0; i < data.length; i++) data[i] = rng.nextDouble();
        return new Tensor(data, shape);
    }

    public static Tensor random(int... shape) {
        return random(new Shape(shape));
    }

    public static Tensor randomNormal(Shape shape, double mean, double std) {
        Random rng = new Random();
        double[] data = new double[shape.totalSize()];
        for (int i = 0; i < data.length; i++) {
            data[i] = rng.nextGaussian() * std + mean;
        }
        return new Tensor(data, shape);
    }

    public static Tensor randomNormal(int... shape) {
        return randomNormal(new Shape(shape), 0, 1);
    }

    public static Tensor randomNormal(Shape shape) {
        return randomNormal(shape, 0, 1);
    }

    public static Tensor linspace(double start, double end, int n) {
        double[] data = new double[n];
        if (n == 1) {
            data[0] = start;
        } else {
            double step = (end - start) / (n - 1);
            for (int i = 0; i < n; i++) data[i] = start + i * step;
        }
        return new Tensor(data, new Shape(n));
    }

    public static Tensor arange(double start, double stop, double step) {
        int n = (int) Math.ceil((stop - start) / step);
        double[] data = new double[n];
        for (int i = 0; i < n; i++) data[i] = start + i * step;
        return new Tensor(data, new Shape(n));
    }

    public static Tensor arange(double stop) {
        return arange(0, stop, 1);
    }

    public static Tensor eye(int n) {
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) data[i * n + i] = 1.0;
        return new Tensor(data, new Shape(n, n));
    }

    public Tensor add(Tensor other) { return elementWise(other, Double::sum); }

    public Tensor add(double scalar) { return elementWiseScalar(d -> d + scalar); }

    public Tensor sub(Tensor other) { return elementWise(other, (a, b) -> a - b); }

    public Tensor sub(double scalar) { return elementWiseScalar(d -> d - scalar); }

    public Tensor mul(Tensor other) { return elementWise(other, (a, b) -> a * b); }

    public Tensor mul(double scalar) { return elementWiseScalar(d -> d * scalar); }

    public Tensor div(Tensor other) { return elementWise(other, (a, b) -> a / b); }

    public Tensor div(double scalar) { return elementWiseScalar(d -> d / scalar); }

    public Tensor pow(double exponent) { return elementWiseScalar(d -> Math.pow(d, exponent)); }

    public Tensor sqrt() { return elementWiseScalar(Math::sqrt); }

    public Tensor exp() { return elementWiseScalar(Math::exp); }

    public Tensor log() { return elementWiseScalar(Math::log); }

    public Tensor abs() { return elementWiseScalar(Math::abs); }

    public Tensor neg() { return elementWiseScalar(d -> -d); }

    public Tensor sigmoid() { return elementWiseScalar(d -> 1.0 / (1.0 + Math.exp(-d))); }

    public Tensor relu() { return elementWiseScalar(d -> Math.max(0, d)); }

    public Tensor tanh() { return elementWiseScalar(Math::tanh); }

    public Tensor softmax() {
        double max = Double.NEGATIVE_INFINITY;
        for (double d : data) if (d > max) max = d;
        double sum = 0;
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = Math.exp(data[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < data.length; i++) result[i] /= sum;
        return new Tensor(result, shape);
    }

    public Tensor leakyRelu(double alpha) {
        return elementWiseScalar(d -> d >= 0 ? d : alpha * d);
    }

    public Tensor leakyRelu() { return leakyRelu(0.01); }

    public Tensor elu(double alpha) {
        return elementWiseScalar(d -> d >= 0 ? d : alpha * (Math.exp(d) - 1));
    }

    public Tensor elu() { return elu(1.0); }

    public Tensor matmul(Tensor other) {
        int[] aDims = this.shape.dims();
        int[] bDims = other.shape.dims();
        if (aDims.length < 2 || bDims.length < 2) {
            throw new IllegalArgumentException("matmul requires at least 2D tensors");
        }
        if (aDims[aDims.length - 1] != bDims[bDims.length - 2]) {
            throw new IllegalArgumentException(
                "Incompatible matrix dimensions: " + Arrays.toString(aDims) + " and " + Arrays.toString(bDims));
        }
        int m = aDims[aDims.length - 2];
        int k = aDims[aDims.length - 1];
        int n = bDims[bDims.length - 1];
        int batchDims = aDims.length - 2;
        int aBatch = 1, bBatch = 1;
        for (int i = 0; i < batchDims; i++) {
            aBatch *= aDims[i];
            bBatch *= bDims[i];
        }
        int maxBatch = Math.max(aBatch, bBatch);
        int[] resultDims = new int[aDims.length];
        for (int i = 0; i < batchDims; i++) {
            resultDims[i] = Math.max(aDims[i], bDims[i]);
        }
        resultDims[batchDims] = m;
        resultDims[batchDims + 1] = n;
        double[] result = new double[maxBatch * m * n];
        for (int b = 0; b < maxBatch; b++) {
            int aOff = (aBatch == 1 ? 0 : b) * m * k;
            int bOff = (bBatch == 1 ? 0 : b) * k * n;
            int rOff = b * m * n;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int t = 0; t < k; t++) {
                        sum += this.data[aOff + i * k + t] * other.data[bOff + t * n + j];
                    }
                    result[rOff + i * n + j] = sum;
                }
            }
        }
        return new Tensor(result, new Shape(resultDims));
    }

    public Tensor transpose() {
        int[] dims = shape.dims();
        if (dims.length < 2) return new Tensor(this);
        int[] perm = new int[dims.length];
        for (int i = 0; i < dims.length - 2; i++) perm[i] = i;
        perm[dims.length - 2] = dims.length - 1;
        perm[dims.length - 1] = dims.length - 2;
        return permute(perm);
    }

    public Tensor permute(int... axes) {
        int[] dims = shape.dims();
        int[] newDims = new int[dims.length];
        for (int i = 0; i < axes.length; i++) newDims[i] = dims[axes[i]];
        Shape newShape = new Shape(newDims);
        double[] newData = new double[data.length];
        int[] strides = new int[dims.length];
        strides[dims.length - 1] = 1;
        for (int i = dims.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        int[] newStrides = new int[dims.length];
        newStrides[dims.length - 1] = 1;
        for (int i = dims.length - 2; i >= 0; i--) {
            newStrides[i] = newStrides[i + 1] * newDims[i + 1];
        }
        int[] indices = new int[dims.length];
        for (int flat = 0; flat < data.length; flat++) {
            int temp = flat;
            for (int i = dims.length - 1; i >= 0; i--) {
                indices[i] = temp % dims[i];
                temp /= dims[i];
            }
            int newFlat = 0;
            for (int i = 0; i < axes.length; i++) {
                newFlat += indices[axes[i]] * newStrides[i];
            }
            newData[newFlat] = data[flat];
        }
        return new Tensor(newData, newShape);
    }

    public Tensor sum() {
        double total = 0;
        for (double d : data) total += d;
        return new Tensor(total);
    }

    public Tensor sum(int axis) {
        int[] dims = shape.dims();
        if (axis < 0) axis += dims.length;
        int[] newDims = new int[dims.length - 1];
        int idx = 0;
        for (int i = 0; i < dims.length; i++) {
            if (i != axis) newDims[idx++] = dims[i];
        }
        if (newDims.length == 0) return sum();
        Shape newShape = new Shape(newDims);
        double[] result = new double[newShape.totalSize()];
        int outer = 1, inner = 1, axisSize = dims[axis];
        for (int i = 0; i < axis; i++) outer *= dims[i];
        for (int i = axis + 1; i < dims.length; i++) inner *= dims[i];
        for (int o = 0; o < outer; o++) {
            for (int a = 0; a < axisSize; a++) {
                for (int inn = 0; inn < inner; inn++) {
                    int srcIdx = o * axisSize * inner + a * inner + inn;
                    int dstIdx = o * inner + inn;
                    result[dstIdx] += data[srcIdx];
                }
            }
        }
        return new Tensor(result, newShape);
    }

    public Tensor mean() {
        return sum().div(data.length);
    }

    public Tensor mean(int axis) {
        Tensor s = sum(axis);
        return s.div(shape.dim(axis < 0 ? axis + shape.numDims() : axis));
    }

    public Tensor min() {
        double m = Double.POSITIVE_INFINITY;
        for (double d : data) if (d < m) m = d;
        return new Tensor(m);
    }

    public Tensor max() {
        double m = Double.NEGATIVE_INFINITY;
        for (double d : data) if (d > m) m = d;
        return new Tensor(m);
    }

    public Tensor min(int axis) {
        return reduceAxis(axis, true);
    }

    public Tensor max(int axis) {
        return reduceAxis(axis, false);
    }

    private Tensor reduceAxis(int axis, boolean isMin) {
        int[] dims = shape.dims();
        if (axis < 0) axis += dims.length;
        int[] newDims = new int[dims.length - 1];
        int idx = 0;
        for (int i = 0; i < dims.length; i++) {
            if (i != axis) newDims[idx++] = dims[i];
        }
        if (newDims.length == 0) return isMin ? min() : max();
        Shape newShape = new Shape(newDims);
        double[] result = new double[newShape.totalSize()];
        Arrays.fill(result, isMin ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
        int outer = 1, inner = 1, axisSize = dims[axis];
        for (int i = 0; i < axis; i++) outer *= dims[i];
        for (int i = axis + 1; i < dims.length; i++) inner *= dims[i];
        for (int o = 0; o < outer; o++) {
            for (int a = 0; a < axisSize; a++) {
                for (int inn = 0; inn < inner; inn++) {
                    int srcIdx = o * axisSize * inner + a * inner + inn;
                    int dstIdx = o * inner + inn;
                    if (isMin) {
                        if (data[srcIdx] < result[dstIdx]) result[dstIdx] = data[srcIdx];
                    } else {
                        if (data[srcIdx] > result[dstIdx]) result[dstIdx] = data[srcIdx];
                    }
                }
            }
        }
        return new Tensor(result, newShape);
    }

    public Tensor argmax() {
        int idx = 0;
        for (int i = 1; i < data.length; i++) {
            if (data[i] > data[idx]) idx = i;
        }
        return new Tensor(idx);
    }

    public Tensor argmax(int axis) {
        int[] dims = shape.dims();
        if (axis < 0) axis += dims.length;
        int[] newDims = new int[dims.length - 1];
        int idx2 = 0;
        for (int i = 0; i < dims.length; i++) {
            if (i != axis) newDims[idx2++] = dims[i];
        }
        if (newDims.length == 0) return argmax();
        Shape newShape = new Shape(newDims);
        double[] result = new double[newShape.totalSize()];
        int outer = 1, inner = 1, axisSize = dims[axis];
        for (int i = 0; i < axis; i++) outer *= dims[i];
        for (int i = axis + 1; i < dims.length; i++) inner *= dims[i];
        for (int o = 0; o < outer; o++) {
            for (int inn = 0; inn < inner; inn++) {
                int bestIdx = 0;
                double bestVal = Double.NEGATIVE_INFINITY;
                for (int a = 0; a < axisSize; a++) {
                    int srcIdx = o * axisSize * inner + a * inner + inn;
                    if (data[srcIdx] > bestVal) {
                        bestVal = data[srcIdx];
                        bestIdx = a;
                    }
                }
                result[o * inner + inn] = bestIdx;
            }
        }
        return new Tensor(result, newShape);
    }

    public Tensor argmin() {
        int idx = 0;
        for (int i = 1; i < data.length; i++) {
            if (data[i] < data[idx]) idx = i;
        }
        return new Tensor(idx);
    }

    public Tensor argmin(int axis) {
        int[] dims = shape.dims();
        if (axis < 0) axis += dims.length;
        int[] newDims = new int[dims.length - 1];
        int idx2 = 0;
        for (int i = 0; i < dims.length; i++) {
            if (i != axis) newDims[idx2++] = dims[i];
        }
        if (newDims.length == 0) return argmin();
        Shape newShape = new Shape(newDims);
        double[] result = new double[newShape.totalSize()];
        int outer = 1, inner = 1, axisSize = dims[axis];
        for (int i = 0; i < axis; i++) outer *= dims[i];
        for (int i = axis + 1; i < dims.length; i++) inner *= dims[i];
        for (int o = 0; o < outer; o++) {
            for (int inn = 0; inn < inner; inn++) {
                int bestIdx = 0;
                double bestVal = Double.POSITIVE_INFINITY;
                for (int a = 0; a < axisSize; a++) {
                    int srcIdx = o * axisSize * inner + a * inner + inn;
                    if (data[srcIdx] < bestVal) {
                        bestVal = data[srcIdx];
                        bestIdx = a;
                    }
                }
                result[o * inner + inn] = bestIdx;
            }
        }
        return new Tensor(result, newShape);
    }

    public Tensor reshape(int... newDims) {
        return reshape(new Shape(newDims));
    }

    public Tensor reshape(Shape newShape) {
        if (newShape.totalSize() != shape.totalSize()) {
            int autoIdx = -1;
            int[] dims = newShape.dims();
            int known = 1;
            for (int i = 0; i < dims.length; i++) {
                if (dims[i] == -1) {
                    if (autoIdx >= 0) throw new IllegalArgumentException("Only one -1 allowed in reshape");
                    autoIdx = i;
                } else {
                    known *= dims[i];
                }
            }
            if (autoIdx >= 0) {
                dims[autoIdx] = shape.totalSize() / known;
                newShape = new Shape(dims);
            } else {
                throw new IllegalArgumentException(
                    "Cannot reshape " + shape + " to " + newShape + " (total sizes differ)");
            }
        }
        return new Tensor(data, newShape);
    }

    public Tensor flatten() {
        return reshape(-1);
    }

    public Tensor expandDims(int axis) {
        return new Tensor(data, shape.expandDims(axis));
    }

    public Tensor squeeze() {
        return new Tensor(data, shape.squeeze());
    }

    public Tensor elementWise(Tensor other, DoubleBinaryOperator op) {
        Shape resultShape = shape.broadcast(other.shape);
        double[] result = new double[resultShape.totalSize()];
        int[] aStrides = computeBroadcastStrides(shape, resultShape);
        int[] bStrides = computeBroadcastStrides(other.shape, resultShape);
        for (int i = 0; i < result.length; i++) {
            int[] indices = resultShape.indicesOf(i);
            int aIdx = 0, bIdx = 0;
            for (int d = 0; d < indices.length; d++) {
                aIdx += (indices[d] % shape.dim(d < shape.numDims() ? d : shape.numDims() - 1)) * aStrides[d];
                bIdx += (indices[d] % other.shape.dim(d < other.shape.numDims() ? d : other.shape.numDims() - 1)) * bStrides[d];
            }
            result[i] = op.applyAsDouble(data[aIdx], other.data[bIdx]);
        }
        return new Tensor(result, resultShape);
    }

    public Tensor elementWiseScalar(DoubleUnaryOperator op) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) result[i] = op.applyAsDouble(data[i]);
        return new Tensor(result, shape);
    }

    private int[] computeBroadcastStrides(Shape s, Shape target) {
        int[] strides = new int[target.numDims()];
        int stride = 1;
        for (int i = s.numDims() - 1; i >= 0; i--) {
            strides[i + target.numDims() - s.numDims()] = stride;
            stride *= s.dim(i);
        }
        for (int i = target.numDims() - s.numDims() - 1; i >= 0; i--) {
            strides[i] = 0;
        }
        return strides;
    }

    public double[][] to2D() {
        if (shape.numDims() != 2) throw new IllegalStateException("Tensor is not 2D: " + shape);
        int rows = shape.dim(0), cols = shape.dim(1);
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data, i * cols, result[i], 0, cols);
        }
        return result;
    }

    public double[] to1D() {
        if (shape.numDims() != 1) throw new IllegalStateException("Tensor is not 1D: " + shape);
        return data.clone();
    }

    public double[][][] to3D() {
        if (shape.numDims() != 3) throw new IllegalStateException("Tensor is not 3D: " + shape);
        int d1 = shape.dim(0), d2 = shape.dim(1), d3 = shape.dim(2);
        double[][][] result = new double[d1][d2][d3];
        int idx = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                System.arraycopy(data, idx, result[i][j], 0, d3);
                idx += d3;
            }
        }
        return result;
    }

    public Tensor copy() {
        return new Tensor(this);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=").append(shape).append(", data=");
        if (shape.totalSize() <= 20) {
            sb.append(Arrays.toString(data));
        } else {
            sb.append("[");
            for (int i = 0; i < 10; i++) sb.append(data[i]).append(", ");
            sb.append("..., ");
            for (int i = data.length - 5; i < data.length; i++) sb.append(data[i]).append(i < data.length - 1 ? ", " : "");
            sb.append("]");
        }
        sb.append(")");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor)) return false;
        Tensor tensor = (Tensor) o;
        return shape.equals(tensor.shape) && Arrays.equals(data, tensor.data);
    }

    @Override
    public int hashCode() {
        int result = shape.hashCode();
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }
}
