package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.core.TensorMath;
import com.tenzor.core.Shape;
import java.util.ArrayList;
import java.util.List;

public class Conv2D extends Layer {
    private int filters;
    private int kernelHeight;
    private int kernelWidth;
    private String padding;
    private int stride;
    private String activation;
    private Variable kernel;
    private Variable bias;
    private Tensor inputCache;
    private int padH, padW;

    public Conv2D(int filters, int kernelSize) {
        this(filters, kernelSize, kernelSize, "valid", 1, "linear");
    }

    public Conv2D(int filters, int kernelHeight, int kernelWidth, String padding, int stride, String activation) {
        super("conv2d_" + filters);
        this.filters = filters;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.padding = padding;
        this.stride = stride;
        this.activation = activation;
    }

    private void initWeights(int inputChannels) {
        if (kernel != null) return;
        int[] kShape = new int[]{kernelHeight, kernelWidth, inputChannels, filters};
        kernel = Variable.randomNormal(kShape);
        kernel.setName(name + "_kernel");
        bias = new Variable(Tensor.zeros(filters));
        bias.setName(name + "_bias");
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        int[] dims = input.shape().dims();
        int batchSize = dims[0];
        int height = dims[1];
        int width = dims[2];
        int channels = dims[3];

        initWeights(channels);

        if (padding.equals("same")) {
            padH = kernelHeight / 2;
            padW = kernelWidth / 2;
        } else {
            padH = 0;
            padW = 0;
        }

        int outH = (height + 2 * padH - kernelHeight) / stride + 1;
        int outW = (width + 2 * padW - kernelWidth) / stride + 1;

        Tensor padded = padInput(input, padH, padW);
        double[] paddedData = padded.getData();
        int paddedH = height + 2 * padH;
        int paddedW = width + 2 * padW;

        double[] kData = kernel.getData();
        double[] bData = bias.getData();
        double[] outData = new double[batchSize * outH * outW * filters];

        for (int n = 0; n < batchSize; n++) {
            for (int f = 0; f < filters; f++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double sum = bData[f];
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                for (int c = 0; c < channels; c++) {
                                    int ih = oh * stride + kh;
                                    int iw = ow * stride + kw;
                                    int inIdx = ((n * paddedH + ih) * paddedW + iw) * channels + c;
                                    int kIdx = ((kh * kernelWidth + kw) * channels + c) * filters + f;
                                    sum += paddedData[inIdx] * kData[kIdx];
                                }
                            }
                        }
                        int outIdx = ((n * outH + oh) * outW + ow) * filters + f;
                        outData[outIdx] = sum;
                    }
                }
            }
        }

        Tensor convOut = new Tensor(outData, new Shape(batchSize, outH, outW, filters));
        this.inputCache = input;
        this.output = com.tenzor.nn.activations.Activations.get(activation, convOut);
        return this.output;
    }

    private Tensor padInput(Tensor input, int padH, int padW) {
        if (padH == 0 && padW == 0) return input;
        return TensorMath.pad(input, new int[]{0, padH, padW, 0}, new int[]{0, padH, padW, 0}, 0.0);
    }

    @Override
    public List<Variable> getParameters() {
        List<Variable> params = new ArrayList<>();
        params.add(kernel);
        params.add(bias);
        return params;
    }

    @Override
    public List<Tensor> getGradients() {
        return new ArrayList<>();
    }

    @Override
    public int[] getOutputShape(int[] inputShape) {
        int h = inputShape[1], w = inputShape[2];
        int outH = (h + 2 * padH - kernelHeight) / stride + 1;
        int outW = (w + 2 * padW - kernelWidth) / stride + 1;
        return new int[]{inputShape[0], outH, outW, filters};
    }
}
