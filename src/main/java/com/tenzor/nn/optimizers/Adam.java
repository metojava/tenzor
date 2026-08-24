package com.tenzor.nn.optimizers;

import com.tenzor.core.Variable;
import com.tenzor.core.Tensor;
import java.util.List;
import java.util.ArrayList;

public class Adam extends Optimizer {
    private double beta1;
    private double beta2;
    private double epsilon;
    private List<Tensor> m;
    private List<Tensor> v;
    private int t;

    public Adam() { this(0.001, 0.9, 0.999, 1e-8); }

    public Adam(double learningRate) { this(learningRate, 0.9, 0.999, 1e-8); }

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
    }

    @Override
    public void update(List<Variable> parameters, List<Tensor> gradients) {
        t++;
        double lrT = learningRate * Math.sqrt(1 - Math.pow(beta2, t)) / (1 - Math.pow(beta1, t));

        if (m == null) {
            m = new ArrayList<>();
            v = new ArrayList<>();
            for (Variable p : parameters) {
                m.add(Tensor.zeros(p.shape()));
                v.add(Tensor.zeros(p.shape()));
            }
        }

        for (int i = 0; i < parameters.size(); i++) {
            Variable param = parameters.get(i);
            Tensor grad = gradients.get(i);
            if (!param.isTrainable()) continue;

            double[] pData = param.getData();
            double[] gData = grad.getData();
            double[] mData = m.get(i).getData();
            double[] vData = v.get(i).getData();

            for (int j = 0; j < pData.length; j++) {
                mData[j] = beta1 * mData[j] + (1 - beta1) * gData[j];
                vData[j] = beta2 * vData[j] + (1 - beta2) * gData[j] * gData[j];
                pData[j] -= lrT * mData[j] / (Math.sqrt(vData[j]) + epsilon);
            }
        }
    }
}
