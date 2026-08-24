# Tenzor - Java Machine Learning Library

A comprehensive Java machine learning library with TensorFlow-like and scikit-learn-like APIs.

## Features

### Core Tensor Operations
- N-dimensional array operations
- Broadcasting support
- Math operations: add, sub, mul, div, pow, sqrt, exp, log
- Activation functions: sigmoid, relu, tanh, softmax, leakyRelu
- Matrix operations: matmul, transpose, reshape
- Reductions: sum, mean, min, max, argmax, argmin

### Neural Network API (TensorFlow Keras-style)
- **Layers**: Dense, Conv2D, MaxPooling2D, LSTM, GRU, SimpleRNN, Dropout, BatchNormalization, Flatten, Embedding, Activation
- **Losses**: MSE, MAE, CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy, Huber, Hinge
- **Optimizers**: SGD, Adam, RMSprop, Adagrad, Adadelta
- **Models**: Sequential and Functional API

### scikit-learn-like API
- **Linear Models**: LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
- **Ensemble**: RandomForestClassifier, RandomForestRegressor, GradientBoosting, AdaBoost, VotingClassifier
- **Trees**: DecisionTreeClassifier, DecisionTreeRegressor
- **SVM**: SVC, SVR, LinearSVC, LinearSVR
- **Neighbors**: KNeighborsClassifier, KNeighborsRegressor
- **Decomposition**: PCA, IncrementalPCA
- **Preprocessing**: StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
- **Model Selection**: train_test_split, KFold, StratifiedKFold, GridSearchCV, cross_val_score

### Data Utilities
- Dataset loading and batching
- Built-in dataset downloaders: MNIST, FashionMNIST, CIFAR-10, CIFAR-100, BostonHousing, Iris

### Time Series Analysis
- Sequence generation
- Technical indicators: RSI, MACD, Bollinger Bands, MA, EMA
- Stock data loading

### Visualization & Charting
- Training history plots
- Predictions vs actual comparisons
- Time series visualization
- Bar charts for model comparison
- Scatter plots
- Histograms
- Confusion matrix visualization
- Residual plots

### Execution Matrix & Training Tracking
- Detailed epoch-by-epoch tracking
- Loss and accuracy metrics
- Training time measurement
- CSV export
- Best metrics tracking

### Model Persistence
- Save/load models to JSON
- Training checkpoints
- Early stopping

### Metrics
- Classification: Accuracy, Precision, Recall, F1-Score, ConfusionMatrix, AUC
- Regression: MSE, MAE, MAPE, R²-Score

## Quick Start

### Maven Dependency
```xml
<dependency>
    <groupId>com.tenzor</groupId>
    <artifactId>tenzor</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Neural Network Example
```java
import com.tenzor.core.Tensor;
import com.tenzor.nn.Sequential;
import com.tenzor.nn.layers.*;
import com.tenzor.nn.losses.*;
import com.tenzor.nn.optimizers.*;
import com.tenzor.training.ExecutionMatrix;
import com.tenzor.visualization.ChartUtils;

// Build model
Sequential model = new Sequential();
model.add(new Dense(128, "relu"));
model.add(new Dropout(0.2));
model.add(new Dense(64, "relu"));
model.add(new Dense(10, "softmax"));

// Compile
model.compile(new SparseCategoricalCrossentropy(), new Adam(0.001));

// Train with execution matrix
ExecutionMatrix execMatrix = model.fit(xTrain, yTrain, 50, 64, 0.001, 0.15);

// Print detailed summary
execMatrix.printSummary();
execMatrix.printEpochTable();

// Generate charts
ChartUtils.plotTrainingHistory(execMatrix.toHistoryMap(), 
    "Training History", "training_history.png");

// Evaluate
model.printEvaluation(xTest, yTest);
```

### scikit-learn Example
```java
import com.tenzor.core.Tensor;
import com.tenzor.sklearn.ensemble.RandomForestClassifier;
import com.tenzor.sklearn.preprocessing.StandardScaler;
import com.tenzor.sklearn.model_selection.train_test_split;
import com.tenzor.metrics.Accuracy;
import com.tenzor.visualization.ChartUtils;

// Split data
Tensor[] split = train_test_split.split(X, y, 0.3, 42);

// Preprocess
StandardScaler scaler = new StandardScaler();
Tensor X_train = scaler.fitTransform(split[0]);
Tensor X_test = scaler.transform(split[1]);

// Train
RandomForestClassifier rf = new RandomForestClassifier(100, 10, 2);
rf.fit(X_train, split[2]);

// Predict
Tensor predictions = rf.predict(X_test);
double accuracy = Accuracy.compute(split[3], predictions);
```

### Fashion MNIST Example
```java
import com.tenzor.datasets.FashionMNIST;
import com.tenzor.core.Tensor;

// Auto-downloads and extracts dataset
FashionMNIST dataset = new FashionMNIST();
Tensor xTrain = dataset.getXTrain();  // 60000 x 784
Tensor yTrain = dataset.getYTrain();  // 60000
Tensor xTest = dataset.getXTest();   // 10000 x 784
Tensor yTest = dataset.getYTest();   // 10000
```

### Time Series Forecasting
```java
import com.tenzor.timeseries.*;
import com.tenzor.core.Tensor;

// Generate or load data
double[] prices = loadStockData();

// Compute technical indicators
double[] sma = TechnicalIndicators.simpleMovingAverage(prices, 20);
double[] ema = TechnicalIndicators.exponentialMovingAverage(prices, 20);
double[] rsi = TechnicalIndicators.rsi(prices, 14);
double[] macd = TechnicalIndicators.macd(prices, 12, 26, 9);
double[][] bollinger = TechnicalIndicators.bollingerBands(prices, 20, 2);

// Create sequences
TimeSeriesDataset dataset = new TimeSeriesDataset(prices, 30, 1);
Tensor[] split = dataset.trainTestSplit(0.8);
```

## Project Structure
```
tenzor/
├── src/main/java/com/tenzor/
│   ├── core/              # Tensor, Variable, Shape, TensorMath
│   ├── nn/                # Neural network API
│   │   ├── layers/        # Layer implementations
│   │   ├── losses/        # Loss functions
│   │   └── optimizers/    # Optimizers
│   ├── sklearn/             # scikit-learn like API
│   │   ├── base/            # Base classes
│   │   ├── linear_model/    # Linear models
│   │   ├── ensemble/        # Ensemble methods
│   │   ├── tree/           # Decision trees
│   │   ├── svm/             # SVM
│   │   ├── neighbors/       # KNN
│   │   ├── decomposition/  # PCA
│   │   ├── preprocessing/   # Scalers, encoders
│   │   └── model_selection/ # train_test_split, KFold
│   ├── datasets/            # Built-in datasets
│   ├── timeseries/          # Time series utilities
│   ├── metrics/             # Evaluation metrics
│   ├── visualization/        # Charting (JFreeChart)
│   ├── training/            # Execution matrix
│   ├── io/                  # Model persistence
│   ├── callbacks/           # Training callbacks
│   └── data/              # Data utilities
├── examples/                 # Runable examples
├── docs/                    # Documentation
└── src/test/               # JUnit tests
```

## Building
```bash
mvn clean package
mvn test
```

## License
MIT License
