package com.tenzor.nn.optimizers;

import com.tenzor.core.Variable;
import com.tenzor.core.Tensor;
import java.util.List;

public abstract class Optimizer {
    protected String name;
    protected double learningRate;

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
        this.name = getClass().getSimpleName();
    }

    public abstract void update(List<Variable> parameters, List<Tensor> gradients);

    public void setLearningRate(double lr) { this.learningRate = lr; }

    public double getLearningRate() { return learningRate; }

    public String getName() { return name; }
}
