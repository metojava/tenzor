# TensorFlow-like API Guide

## Core Tensor Operations

```java
// Create tensors
Tensor t1 = new Tensor(new double[]{1, 2, 3});
Tensor t2 = Tensor.zeros(3, 4);
Tensor t3 = Tensor.ones(2, 3);
Tensor t4 = Tensor.random(5, 5);
Tensor t5 = Tensor.randomNormal(100);
Tensor t6 = Tensor.linspace(0, 1, 10);
Tensor t7 = Tensor.eye(5);

// Math operations
Tensor sum = t1.add(t2);
Tensor product = t1.mul(t2);
Tensor matMul = t1.matmul(t2.transpose());

// Activation functions
Tensor s = t1.sigmoid();
Tensor r = t1.relu();
Tensor sm = t1.softmax();

// Reduction
Tensor total = t1.sum();
Tensor avg = t1.mean(0);
Tensor maxIdx = t1.argmax();

// Shape operations
Tensor flat = t1.flatten();
Tensor expanded = t1.expandDims(0);
```

## Building Models (Sequential API)

```java
Sequential model = new Sequential();
model.add(new Dense(128, "relu"));
model.add(new Dropout(0.5));
model.add(new Dense(64, "relu"));
model.add(new Dense(10, "softmax"));
model.compile(new CategoricalCrossentropy(), new Adam(0.001));
model.fit(xTrain, yTrain, 10, 32, 0.001);
```

## Available Layers

- **Dense** - Fully connected layer
- **Conv2D** - 2D convolution
- **MaxPooling2D** - Max pooling
- **AveragePooling2D** - Average pooling
- **Flatten** - Flatten input
- **Dropout** - Dropout regularization
- **BatchNormalization** - Batch normalization
- **LSTM** - Long Short-Term Memory
- **GRU** - Gated Recurrent Unit
- **SimpleRNN** - Simple RNN
- **Embedding** - Word embeddings
- **Activation** - Activation function layer

## Loss Functions

- MeanSquaredError (MSE)
- MeanAbsoluteError (MAE)
- CategoricalCrossentropy
- SparseCategoricalCrossentropy
- BinaryCrossentropy
- Huber
- Hinge

## Optimizers

- SGD (with momentum)
- Adam
- RMSprop
- Adagrad
- Adadelta
