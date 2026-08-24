package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.core.Shape;
import java.util.ArrayList;
import java.util.List;

public class AveragePooling2D extends Layer {
    private int poolHeight;
    private int poolWidth;
    private int stride;

    public AveragePooling2D(int poolSize) {
        this(poolSize, poolSize, poolSize);
    }

    public AveragePooling2D(int poolHeight, int poolWidth, int stride) {
        super("avgpool2d");
        this.poolHeight = poolHeight;
        this.poolWidth = poolWidth;
        this.stride = stride;
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        int[] dims = input.shape().dims();
        int batchSize = dims[0];
        int height = dims[1];
        int width = dims[2];
        int channels = dims[3];

        int outH = (height - poolHeight) / stride + 1;
        int outW = (width - poolWidth) / stride + 1;

        double[] inData = input.getData();
        double[] outData = new double[batchSize * outH * outW * channels];
        double poolSize = poolHeight * poolWidth;

        for (int n = 0; n < batchSize; n++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double sum = 0;
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                int idx = ((n * height + ih) * width + iw) * channels + c;
                                sum += inData[idx];
                            }
                        }
                        int outIdx = ((n * outH + oh) * outW + ow) * channels + c;
                        outData[outIdx] = sum / poolSize;
                    }
                }
            }
        }

        this.output = new Tensor(outData, new Shape(batchSize, outH, outW, channels));
        return output;
    }

    @Override
    public List<Variable> getParameters() {
        return new ArrayList<>();
    }

    @Override
    public List<Tensor> getGradients() {
        return new ArrayList<>();
    }

    @Override
    public int[] getOutputShape(int[] inputShape) {
        int h = inputShape[1], w = inputShape[2];
        int outH = (h - poolHeight) / stride + 1;
        int outW = (w - poolWidth) / stride + 1;
        return new int[]{inputShape[0], outH, outW, inputShape[3]};
    }
}
