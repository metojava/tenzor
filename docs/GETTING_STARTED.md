# Getting Started with Tenzor

## Prerequisites

- Java 11+
- Maven 3.6+

## Installation

Add to your `pom.xml`:

```xml
<dependency>
    <groupId>com.tenzor</groupId>
    <artifactId>tenzor</artifactId>
    <version>1.0.0</version>
</dependency>
```

Or build from source:

```bash
git clone https://github.com/yourusername/tenzor
cd tenzor
mvn clean install
```

## Your First Neural Network

```java
import com.tenzor.core.Tensor;
import com.tenzor.nn.Sequential;
import com.tenzor.nn.layers.*;
import com.tenzor.nn.losses.MeanSquaredError;
import com.tenzor.nn.optimizers.SGD;

public class FirstNN {
    public static void main(String[] args) {
        // Create training data (XOR problem)
        Tensor x = new Tensor(new double[][]{
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        });
        Tensor y = new Tensor(new double[]{0, 1, 1, 0});

        // Build model
        Sequential model = new Sequential();
        model.add(new Dense(8, "relu"));
        model.add(new Dense(1, "sigmoid"));
        model.compile(new MeanSquaredError(), new SGD(0.1));
        model.fit(x, y, 1000, 4, 0.1);

        // Predict
        Tensor predictions = model.predict(x);
        System.out.println(predictions);
    }
}
```

## Your First scikit-learn Model

```java
import com.tenzor.sklearn.linear_model.LinearRegression;
import com.tenzor.sklearn.model_selection.train_test_split;
import com.tenzor.metrics.R2Score;

// Load data, train, predict
LinearRegression lr = new LinearRegression();
lr.fit(xTrain, yTrain);
Tensor pred = lr.predict(xTest);
double r2 = R2Score.compute(yTest, pred);
```

## Working with Datasets

```java
FashionMNIST dataset = new FashionMNIST();
Tensor xTrain = dataset.getXTrain();
Tensor yTrain = dataset.getYTrain();
```

## Saving and Loading Models

```java
ModelSaver.save(model, "model.json");
Sequential loaded = ModelLoader.load("model.json");
```
