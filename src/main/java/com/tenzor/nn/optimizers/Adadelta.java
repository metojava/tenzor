package com.tenzor.nn.optimizers;

import com.tenzor.core.Variable;
import com.tenzor.core.Tensor;
import java.util.List;
import java.util.ArrayList;

public class Adadelta extends Optimizer {
    private double rho;
    private double epsilon;
    private List<Tensor> egCache;
    private List<Tensor> deltaCache;

    public Adadelta() { this(0.95, 1e-8); }

    public Adadelta(double rho, double epsilon) {
        super(1.0);
        this.rho = rho;
        this.epsilon = epsilon;
    }

    @Override
    public void update(List<Variable> parameters, List<Tensor> gradients) {
        if (egCache == null) {
            egCache = new ArrayList<>();
            deltaCache = new ArrayList<>();
            for (Variable p : parameters) {
                egCache.add(Tensor.zeros(p.shape()));
                deltaCache.add(Tensor.zeros(p.shape()));
            }
        }

        for (int i = 0; i < parameters.size(); i++) {
            Variable param = parameters.get(i);
            Tensor grad = gradients.get(i);
            if (!param.isTrainable()) continue;

            double[] pData = param.getData();
            double[] gData = grad.getData();
            double[] egData = egCache.get(i).getData();
            double[] dxData = deltaCache.get(i).getData();

            for (int j = 0; j < pData.length; j++) {
                egData[j] = rho * egData[j] + (1 - rho) * gData[j] * gData[j];
                double delta = Math.sqrt(dxData[j] + epsilon) / Math.sqrt(egData[j] + epsilon) * gData[j];
                dxData[j] = rho * dxData[j] + (1 - rho) * delta * delta;
                pData[j] -= delta;
            }
        }
    }
}
