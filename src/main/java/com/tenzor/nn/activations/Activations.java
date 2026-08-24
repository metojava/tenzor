package com.tenzor.nn.activations;

import com.tenzor.core.Tensor;

public final class Activations {

    private Activations() {}

    public static Tensor sigmoid(Tensor x) { return x.sigmoid(); }

    public static Tensor relu(Tensor x) { return x.relu(); }

    public static Tensor tanh(Tensor x) { return x.tanh(); }

    public static Tensor softmax(Tensor x) { return x.softmax(); }

    public static Tensor leakyRelu(Tensor x, double alpha) { return x.leakyRelu(alpha); }

    public static Tensor leakyRelu(Tensor x) { return x.leakyRelu(); }

    public static Tensor elu(Tensor x, double alpha) { return x.elu(alpha); }

    public static Tensor elu(Tensor x) { return x.elu(); }

    public static Tensor linear(Tensor x) { return new Tensor(x); }

    public static Tensor swish(Tensor x) {
        return x.mul(x.sigmoid());
    }

    public static Tensor gelu(Tensor x) {
        return x.elementWiseScalar(d -> {
            double c = Math.sqrt(2.0 / Math.PI) * (d + 0.044715 * Math.pow(d, 3));
            return 0.5 * d * (1.0 + Math.tanh(c));
        });
    }

    public static Tensor softplus(Tensor x) {
        return x.elementWiseScalar(d -> Math.log(1.0 + Math.exp(d)));
    }

    public static Tensor softsign(Tensor x) {
        return x.elementWiseScalar(d -> d / (1.0 + Math.abs(d)));
    }

    public static Tensor hardSigmoid(Tensor x) {
        return x.elementWiseScalar(d -> Math.max(0, Math.min(1, 0.2 * d + 0.5)));
    }

    public static Tensor get(String name, Tensor x) {
        switch (name.toLowerCase()) {
            case "sigmoid": return sigmoid(x);
            case "relu": return relu(x);
            case "tanh": return tanh(x);
            case "softmax": return softmax(x);
            case "leakyrelu": return leakyRelu(x);
            case "elu": return elu(x);
            case "linear": return linear(x);
            case "swish": return swish(x);
            case "gelu": return gelu(x);
            case "softplus": return softplus(x);
            case "softsign": return softsign(x);
            case "hardsigmoid": return hardSigmoid(x);
            default: throw new IllegalArgumentException("Unknown activation: " + name);
        }
    }

    public static Tensor derivative(Tensor x, String activation) {
        switch (activation.toLowerCase()) {
            case "sigmoid": {
                Tensor s = sigmoid(x);
                return s.mul(s.neg().add(1.0));
            }
            case "relu":
                return x.elementWiseScalar(d -> d > 0 ? 1.0 : 0.0);
            case "tanh": {
                Tensor t = tanh(x);
                return t.mul(t).neg().add(1.0);
            }
            case "linear":
                return Tensor.ones(x.shape());
            case "leakyrelu":
                return x.elementWiseScalar(d -> d >= 0 ? 1.0 : 0.01);
            case "elu":
                return x.elementWiseScalar(d -> d >= 0 ? 1.0 : Math.exp(d));
            default:
                return Tensor.ones(x.shape());
        }
    }
}
