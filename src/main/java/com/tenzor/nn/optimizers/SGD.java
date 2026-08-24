package com.tenzor.nn.optimizers;

import com.tenzor.core.Variable;
import com.tenzor.core.Tensor;
import java.util.List;

public class SGD extends Optimizer {
    private double momentum;
    private List<Tensor> velocities;

    public SGD() { this(0.01, 0.0); }

    public SGD(double learningRate) { this(learningRate, 0.0); }

    public SGD(double learningRate, double momentum) {
        super(learningRate);
        this.momentum = momentum;
    }

    @Override
    public void update(List<Variable> parameters, List<Tensor> gradients) {
        for (int i = 0; i < parameters.size(); i++) {
            Variable param = parameters.get(i);
            Tensor grad = gradients.get(i);
            if (!param.isTrainable()) continue;

            double[] pData = param.getData();
            double[] gData = grad.getData();

            if (momentum > 0) {
                if (velocities == null || velocities.size() <= i) {
                    velocities = new java.util.ArrayList<>();
                    while (velocities.size() <= i) velocities.add(Tensor.zeros(param.shape()));
                }
                double[] vData = velocities.get(i).getData();
                for (int j = 0; j < pData.length; j++) {
                    vData[j] = momentum * vData[j] + learningRate * gData[j];
                    pData[j] -= vData[j];
                }
            } else {
                for (int j = 0; j < pData.length; j++) {
                    pData[j] -= learningRate * gData[j];
                }
            }
        }
    }
}
