package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import java.util.ArrayList;
import java.util.List;

public class Input extends Layer {
    private int[] inputShape;

    public Input(int... inputShape) {
        super("input");
        this.inputShape = inputShape;
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        this.output = input;
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
        return this.inputShape != null ? this.inputShape : inputShape;
    }

    public int[] getInputShape() { return inputShape; }

    public void setInputShape(int[] inputShape) { this.inputShape = inputShape; }
}
