package com.tenzor.core;

public class Variable extends Tensor {
    private boolean trainable;
    private String name;

    public Variable() {
        super();
        this.trainable = true;
        this.name = "variable";
    }

    public Variable(double value) {
        super(value);
        this.trainable = true;
        this.name = "variable";
    }

    public Variable(double[] data) {
        super(data);
        this.trainable = true;
        this.name = "variable";
    }

    public Variable(double[][] data) {
        super(data);
        this.trainable = true;
        this.name = "variable";
    }

    public Variable(double[][][] data) {
        super(data);
        this.trainable = true;
        this.name = "variable";
    }

    public Variable(double[] data, Shape shape) {
        super(data, shape);
        this.trainable = true;
        this.name = "variable";
    }

    public Variable(Shape shape) {
        super(new double[shape.totalSize()], shape);
        this.trainable = true;
        this.name = "variable";
    }

    public Variable(Tensor tensor) {
        super(tensor);
        this.trainable = true;
        this.name = "variable";
    }

    public boolean isTrainable() { return trainable; }

    public void setTrainable(boolean trainable) { this.trainable = trainable; }

    public String getName() { return name; }

    public void setName(String name) { this.name = name; }

    public static Variable zeros(Shape shape) {
        return new Variable(Tensor.zeros(shape));
    }

    public static Variable zeros(int... shape) {
        return new Variable(Tensor.zeros(shape));
    }

    public static Variable ones(Shape shape) {
        return new Variable(Tensor.ones(shape));
    }

    public static Variable ones(int... shape) {
        return new Variable(Tensor.ones(shape));
    }

    public static Variable random(Shape shape) {
        return new Variable(Tensor.random(shape));
    }

    public static Variable random(int... shape) {
        return new Variable(Tensor.random(shape));
    }

    public static Variable randomNormal(Shape shape, double mean, double std) {
        return new Variable(Tensor.randomNormal(shape, mean, std));
    }

    public static Variable randomNormal(Shape shape) {
        return new Variable(Tensor.randomNormal(shape));
    }

    public static Variable randomNormal(int... shape) {
        return new Variable(Tensor.randomNormal(shape));
    }

    public Variable copy() {
        Variable v = new Variable(this);
        v.trainable = this.trainable;
        v.name = this.name;
        return v;
    }

    public void assign(Tensor other) {
        if (!this.shape.equals(other.shape)) {
            throw new IllegalArgumentException("Shape mismatch for assign: " + this.shape + " vs " + other.shape);
        }
        System.arraycopy(other.data, 0, this.data, 0, this.data.length);
    }

    @Override
    public String toString() {
        return "Variable(name=" + name + ", trainable=" + trainable + ", shape=" + shape + ")";
    }
}
