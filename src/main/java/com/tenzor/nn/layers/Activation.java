package com.tenzor.nn.layers;

import com.tenzor.core.Tensor;
import com.tenzor.core.Variable;
import com.tenzor.nn.activations.Activations;
import java.util.ArrayList;
import java.util.List;

public class Activation extends Layer {
    private String activationName;

    public Activation(String activationName) {
        super("activation_" + activationName);
        this.activationName = activationName;
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        this.output = Activations.get(activationName, input);
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
        return inputShape.clone();
    }

    public String getActivationName() { return activationName; }
}
