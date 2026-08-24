# Spring Boot Starter Tenzor

Spring Boot integration for the Tenzor Machine Learning Library.

## Installation

### Maven
Add this dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>com.tenzor</groupId>
    <artifactId>spring-boot-starter-tenzor</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Gradle
```groovy
implementation 'com.tenzor:spring-boot-starter-tenzor:1.0.0'
```

## Configuration Properties

Configure Tenzor in your `application.properties` or `application.yml`:

```yaml
tenzor:
  model:
    save-path: models/
    load-path: models/my_model.json
  training:
    epochs: 100
    batch-size: 32
    learning-rate: 0.001
    validation-split: 0.15
    early-stopping: true
    early-stopping-patience: 10
    checkpointing: true
    checkpoint-path: checkpoints/
  visualization:
    enabled: true
    output-path: charts/
    chart-width: 1000
    chart-height: 600
```

## Usage

### 1. Basic Neural Network

```java
@Service
public class MyModelService {

    @Autowired
    private TenzorTemplate tenzorTemplate;

    public void trainModel() {
        // Build model using fluent API
        TenzorTemplate.ModelConfig config = new TenzorTemplate.ModelConfig()
            .layer(TenzorTemplate.LayerConfig.dense(128, "relu"))
            .layer(TenzorTemplate.LayerConfig.dropout(0.2))
            .layer(TenzorTemplate.LayerConfig.dense(64, "relu"))
            .layer(TenzorTemplate.LayerConfig.dense(10, "softmax"))
            .loss("sparsecategoricalcrossentropy")
            .optimizer("adam")
            .learningRate(0.001);

        Sequential model = tenzorTemplate.createModel(config);

        // Prepare data
        Tensor xTrain = tenzorTemplate.createTensor(trainData);
        Tensor yTrain = tenzorTemplate.createTensor(trainLabels);

        // Train - auto-configured callbacks and visualization
        ExecutionMatrix history = tenzorTemplate.train(model, xTrain, yTrain);

        // Save model
        tenzorTemplate.saveModel(model, "my_classifier");

        // Predict
        Tensor predictions = tenzorTemplate.predict(model, xTest);
    }
}
```

### 2. Time Series with LSTM

```java
@Service
public class TimeSeriesService {

    @Autowired
    private TenzorTemplate tenzorTemplate;

    @Autowired
    private DataPreparationService dataService;

    public void buildTimeSeriesModel(double[] prices) {
        // Prepare sequence data
        DataPreparationService.TimeSeriesResult result = 
            dataService.prepareTimeSeries(prices, 30, 1);

        // Build LSTM model
        TenzorTemplate.ModelConfig config = new TenzorTemplate.ModelConfig()
            .layer(TenzorTemplate.LayerConfig.lstm(64, true))   // return sequences
            .layer(TenzorTemplate.LayerConfig.dropout(0.2))
            .layer(TenzorTemplate.LayerConfig.lstm(32, false))  // final LSTM
            .layer(TenzorTemplate.LayerConfig.dense(1, "linear"))
            .loss("mse")
            .optimizer("adam");

        Sequential model = tenzorTemplate.createModel(config);

        // Train
        tenzorTemplate.train(model, result.xTrain, result.yTrain, 100, 32, 0.001, 0.1);
    }
}
```

### 3. Model Registry

```java
@RestController
@RequestMapping("/api/models")
public class ModelController {

    @Autowired
    private ModelRegistry modelRegistry;

    @Autowired
    private TenzorTemplate tenzorTemplate;

    @PostMapping("/{name}")
    public void saveModel(@PathVariable String name, @RequestBody Sequential model) {
        ModelRegistry.ModelMetadata meta = new ModelRegistry.ModelMetadata();
        meta.setDescription("Image classifier model");
        meta.setVersion("1.0.0");
        
        modelRegistry.register(name, model, meta);
    }

    @GetMapping("/{name}/predict")
    public Tensor predict(@PathVariable String name, @RequestBody Tensor input) {
        Sequential model = modelRegistry.get(name);
        return model.predict(input);
    }
}
```

### 4. Data Preparation

```java
@Service
public class DataPipeline {

    @Autowired
    private DataPreparationService dataService;

    public void processData(Tensor X, Tensor y) {
        // Train/test split
        Tensor[] split = dataService.trainTestSplit(X, y, 0.3, 42);
        Tensor xTrain = split[0];
        Tensor xTest = split[1];
        Tensor yTrain = split[2];
        Tensor yTest = split[3];

        // Scale features
        DataPreparationService.ScalerResult scaled = dataService.standardScale(xTrain, xTest);
        Tensor xTrainScaled = scaled.trainScaled;
        Tensor xTestScaled = scaled.testScaled;
    }
}
```

## Auto-Configuration

The starter automatically provides:
- `TenzorTemplate` - High-level API for model building and training
- `DataPreparationService` - Data preprocessing utilities
- `ModelRegistry` - In-memory model registry
- `TenzorProperties` - Configuration properties

All beans are auto-configured when Spring Boot detects Tenzor on the classpath.

## Available Layers

Use `TenzorTemplate.LayerConfig` factory methods:

| Method | Description |
|--------|-------------|
| `dense(units, activation)` | Fully connected layer |
| `dropout(rate)` | Dropout regularization |
| `lstm(units, returnSequences)` | LSTM layer |
| `gru(units, returnSequences)` | GRU layer |
| `flatten()` | Flatten to 1D |
| `batchNormalization()` | Batch normalization |
| `conv2d(filters, kernelSize, activation)` | 2D convolution |
| `maxPooling2d(poolSize)` | Max pooling |
| `embedding(vocabSize, dim)` | Embedding layer |

## Available Loss Functions

- `"mse"` / `"meansquarederror"`
- `"mae"` / `"meanabsoluteerror"`
- `"binarycrossentropy"`
- `"categoricalcrossentropy"`
- `"sparsecategoricalcrossentropy"`
- `"huber"`
- `"hinge"`

## Available Optimizers

- `"adam"` - Adam (default)
- `"sgd"` - SGD with momentum
- `"rmsprop"` - RMSprop
- `"adagrad"` - Adagrad
- `"adadelta"` - Adadelta

## Visualization

When `tenzor.visualization.enabled=true` (default):
- Training history plots auto-generated
- Charts saved to `tenzor.visualization.output-path`
- Use `tenzorTemplate.plotPredictions()`, `plotTimeSeries()`, etc.

## License

MIT License
