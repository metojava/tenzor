package com.tenzor.nn.losses;

import com.tenzor.core.Tensor;

public abstract class Loss {
    protected String name;

    public Loss() { this.name = getClass().getSimpleName(); }

    public Loss(String name) { this.name = name; }

    public abstract Tensor forward(Tensor predictions, Tensor targets);

    public abstract Tensor backward(Tensor predictions, Tensor targets);

    public String getName() { return name; }
}
