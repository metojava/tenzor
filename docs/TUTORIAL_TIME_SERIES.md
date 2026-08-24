# Time Series Tutorial

## Generating Sequences

```java
double[] data = SequenceGenerator.generateSineWave(1000, 10, 1.0, 0);

// Normalize
TimeSeriesPreprocessor preprocessor = new TimeSeriesPreprocessor();
data = preprocessor.normalizeZScore(data);

// Create sequences
TimeSeriesDataset tsData = new TimeSeriesDataset(data, 20, 1);
Tensor[] split = tsData.trainTestSplit(0.8);
```

## Building an LSTM Model

```java
Sequential model = new Sequential();
model.add(new LSTM(50, true));
model.add(new Dropout(0.2));
model.add(new Dense(1));
model.compile(new MeanSquaredError(), new Adam(0.01));
model.fit(xTrain, yTrain, 20, 32, 0.01);
```

## Technical Indicators

```java
double[] sma = TechnicalIndicators.simpleMovingAverage(prices, 20);
double[] ema = TechnicalIndicators.exponentialMovingAverage(prices, 12);
double[] rsi = TechnicalIndicators.rsi(prices, 14);
double[][] macd = TechnicalIndicators.macd(prices, 12, 26, 9);
double[][] bollinger = TechnicalIndicators.bollingerBands(prices, 20, 2);
double[] roc = TechnicalIndicators.rateOfChange(prices, 10);
```
