package com.tenzor.nn.optimizers;

import com.tenzor.core.Variable;
import com.tenzor.core.Tensor;
import java.util.List;
import java.util.ArrayList;

public class Adagrad extends Optimizer {
    private double epsilon;
    private List<Tensor> cache;

    public Adagrad() { this(0.01, 1e-8); }

    public Adagrad(double learningRate, double epsilon) {
        super(learningRate);
        this.epsilon = epsilon;
    }

    @Override
    public void update(List<Variable> parameters, List<Tensor> gradients) {
        if (cache == null) {
            cache = new ArrayList<>();
            for (Variable p : parameters) {
                cache.add(Tensor.zeros(p.shape()));
            }
        }

        for (int i = 0; i < parameters.size(); i++) {
            Variable param = parameters.get(i);
            Tensor grad = gradients.get(i);
            if (!param.isTrainable()) continue;

            double[] pData = param.getData();
            double[] gData = grad.getData();
            double[] cData = cache.get(i).getData();

            for (int j = 0; j < pData.length; j++) {
                cData[j] += gData[j] * gData[j];
                pData[j] -= learningRate * gData[j] / (Math.sqrt(cData[j]) + epsilon);
            }
        }
    }
}
