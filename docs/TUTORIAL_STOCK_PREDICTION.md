# Stock Price Prediction Tutorial

## Loading Stock Data

```java
StockDataLoader loader = new StockDataLoader();
double[] prices = loader.loadFromCSV("AAPL.csv", "Close", true);
```

## Computing Technical Indicators

```java
double[] sma20 = TechnicalIndicators.simpleMovingAverage(prices, 20);
double[] ema12 = TechnicalIndicators.exponentialMovingAverage(prices, 12);
double[] rsi = TechnicalIndicators.rsi(prices, 14);
double[][] macd = TechnicalIndicators.macd(prices, 12, 26, 9);
double[][] bands = TechnicalIndicators.bollingerBands(prices, 20, 2);
```

## Building the Feature Set

Combine price, moving averages, RSI, and Bollinger Bands into a multi-dimensional feature set. Normalize and create sequences for LSTM training.

## Training

Use a sequence length of 10-60 days depending on your use case. Train with MSE loss and Adam optimizer.

## Evaluation

Use MAPE (Mean Absolute Percentage Error) for interpretable error metrics.
