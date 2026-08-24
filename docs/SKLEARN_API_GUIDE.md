# scikit-learn API Guide

## Linear Models

```java
LinearRegression lr = new LinearRegression();
lr.fit(xTrain, yTrain);
Tensor pred = lr.predict(xTest);

LogisticRegression logReg = new LogisticRegression(0.01, 1000, 0.0);
logReg.fit(xTrain, yTrain);
Tensor proba = logReg.predictProba(xTest);

Ridge ridge = new Ridge(1.0, true);
Lasso lasso = new Lasso(0.1, true);
ElasticNet en = new ElasticNet(1.0, 0.5, true);
```

## Ensemble Models

```java
RandomForestClassifier rf = new RandomForestClassifier(100, 10, 2);
RandomForestRegressor rfReg = new RandomForestRegressor(100, 10);
GradientBoostingClassifier gbc = new GradientBoostingClassifier(100, 0.1, 3);
AdaBoostClassifier ada = new AdaBoostClassifier(50);

List<Classifier> classifiers = Arrays.asList(rf, gbc);
VotingClassifier voter = new VotingClassifier(classifiers);
```

## SVM

```java
SVC svc = new SVC(1.0, 0.001, 1000, 1e-4);
SVR svr = new SVR(1.0, 0.1, 0.001, 1000);
LinearSVC lsvc = new LinearSVC(1.0, 0.001, 1000);
```

## Tree

```java
DecisionTreeClassifier dt = new DecisionTreeClassifier(10, 2);
DecisionTreeRegressor dtr = new DecisionTreeRegressor(10, 2);
```

## Neighbors

```java
KNeighborsClassifier knn = new KNeighborsClassifier(5);
KNeighborsRegressor knnReg = new KNeighborsRegressor(5);
```

## Decomposition

```java
PCA pca = new PCA(2);
pca.fit(xTrain, null);
Tensor transformed = pca.transform(xTest);
```

## Preprocessing

```java
StandardScaler scaler = new StandardScaler();
MinMaxScaler mmScaler = new MinMaxScaler(0, 1);
RobustScaler rScaler = new RobustScaler();
Normalizer normalizer = new Normalizer("l2");
OneHotEncoder ohe = new OneHotEncoder();
LabelEncoder le = new LabelEncoder();
SimpleImputer imputer = new SimpleImputer("mean", 0);
PolynomialFeatures pf = new PolynomialFeatures(2, true);

Tensor xScaled = scaler.fitTransform(xTrain);
Tensor xTestScaled = scaler.transform(xTest);
```

## Model Selection

```java
Tensor[] split = train_test_split.split(x, y, 0.2, 42);
KFold kfold = new KFold(5, true, 42);
List<int[][]> folds = kfold.split(n);

Map<String, double[]> paramGrid = new HashMap<>();
paramGrid.put("C", new double[]{0.1, 1.0, 10.0});
GridSearchCV gs = new GridSearchCV(estimator, paramGrid, 5);
gs.fit(xTrain, yTrain);

double[] scores = cross_val_score.evaluate(estimator, x, y, 5);
```
