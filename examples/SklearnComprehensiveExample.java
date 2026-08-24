package com.tenzor.examples;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import com.tenzor.sklearn.ensemble.*;
import com.tenzor.sklearn.linear_model.*;
import com.tenzor.sklearn.tree.*;
import com.tenzor.sklearn.svm.*;
import com.tenzor.sklearn.neighbors.*;
import com.tenzor.sklearn.preprocessing.*;
import com.tenzor.sklearn.decomposition.*;
import com.tenzor.sklearn.model_selection.*;
import com.tenzor.metrics.*;
import com.tenzor.datasets.*;
import com.tenzor.visualization.ChartUtils;

import java.util.*;
import java.io.*;
import java.nio.file.*;

public class SklearnComprehensiveExample {
    public static void main(String[] args) throws Exception {
        String outputDir = "output/sklearn_demo";
        Files.createDirectories(Paths.get(outputDir));

        System.out.println("=".repeat(80));
        System.out.println("SCIKIT-LEARN STYLE API DEMONSTRATION");
        System.out.println("=".repeat(80));

        System.out.println("\nLoading Iris dataset...");
        Iris iris = new Iris();
        Tensor X = iris.getFeatures();
        Tensor y = iris.getTarget();

        System.out.println("Features shape: " + X.shape());
        System.out.println("Target shape:   " + y.shape());
        System.out.println("Classes:        " + Arrays.toString(iris.getClassNames()));

        System.out.println("\nSplitting data into train/test sets...");
        Tensor[] split = train_test_split.split(X, y, 0.3, 42);
        Tensor X_train = split[0];
        Tensor X_test = split[1];
        Tensor y_train = split[2];
        Tensor y_test = split[3];

        System.out.println("Train size: " + X_train.shape().dim(0));
        System.out.println("Test size:  " + X_test.shape().dim(0));

        System.out.println("\nPreprocessing data...");
        
        StandardScaler scaler = new StandardScaler();
        Tensor X_train_scaled = scaler.fitTransform(X_train);
        Tensor X_test_scaled = scaler.transform(X_test);

        System.out.println("  - StandardScaler: fit on training data, transform both");

        Map<String, Object> results = new LinkedHashMap<>();
        Map<String, Tensor> predictionsMap = new LinkedHashMap<>();

        System.out.println("\n" + "=".repeat(80));
        System.out.println("TRAINING CLASSIFIERS");
        System.out.println("=".repeat(80));

        System.out.println("\n1. Logistic Regression...");
        LogisticRegression logReg = new LogisticRegression(0.01, 1000);
        logReg.fit(X_train_scaled, y_train);
        Tensor logReg_pred = logReg.predict(X_test_scaled);
        double logReg_acc = Accuracy.compute(y_test, logReg_pred);
        results.put("Logistic Regression", logReg_acc);
        predictionsMap.put("Logistic Regression", logReg_pred);
        System.out.println("   Accuracy: " + String.format("%.4f", logReg_acc));

        System.out.println("\n2. Random Forest Classifier...");
        RandomForestClassifier rf = new RandomForestClassifier(100, 10, 2);
        rf.fit(X_train, y_train);
        Tensor rf_pred = rf.predict(X_test);
        double rf_acc = Accuracy.compute(y_test, rf_pred);
        results.put("Random Forest", rf_acc);
        predictionsMap.put("Random Forest", rf_pred);
        System.out.println("   Accuracy: " + String.format("%.4f", rf_acc));

        System.out.println("\n3. Decision Tree Classifier...");
        DecisionTreeClassifier dt = new DecisionTreeClassifier(10, 2);
        dt.fit(X_train, y_train);
        Tensor dt_pred = dt.predict(X_test);
        double dt_acc = Accuracy.compute(y_test, dt_pred);
        results.put("Decision Tree", dt_acc);
        predictionsMap.put("Decision Tree", dt_pred);
        System.out.println("   Accuracy: " + String.format("%.4f", dt_acc));

        System.out.println("\n4. K-Nearest Neighbors...");
        KNeighborsClassifier knn = new KNeighborsClassifier(5);
        knn.fit(X_train_scaled, y_train);
        Tensor knn_pred = knn.predict(X_test_scaled);
        double knn_acc = Accuracy.compute(y_test, knn_pred);
        results.put("KNN", knn_acc);
        predictionsMap.put("KNN", knn_pred);
        System.out.println("   Accuracy: " + String.format("%.4f", knn_acc));

        System.out.println("\n5. SVM Classifier...");
        SVC svc = new SVC(1.0, "rbf");
        svc.fit(X_train_scaled, y_train);
        Tensor svc_pred = svc.predict(X_test_scaled);
        double svc_acc = Accuracy.compute(y_test, svc_pred);
        results.put("SVM", svc_acc);
        predictionsMap.put("SVM", svc_pred);
        System.out.println("   Accuracy: " + String.format("%.4f", svc_acc));

        System.out.println("\n6. AdaBoost Classifier...");
        AdaBoostClassifier ada = new AdaBoostClassifier(50);
        ada.fit(X_train, y_train);
        Tensor ada_pred = ada.predict(X_test);
        double ada_acc = Accuracy.compute(y_test, ada_pred);
        results.put("AdaBoost", ada_acc);
        predictionsMap.put("AdaBoost", ada_pred);
        System.out.println("   Accuracy: " + String.format("%.4f", ada_acc));

        System.out.println("\n" + "=".repeat(80));
        System.out.println("CLASSIFIER COMPARISON");
        System.out.println("=".repeat(80));
        
        List<Map.Entry<String, Object>> sortedResults = new ArrayList<>(results.entrySet());
        sortedResults.sort((a, b) -> Double.compare((Double) b.getValue(), (Double) a.getValue()));

        System.out.printf("%-25s | %-10s%n", "Classifier", "Accuracy");
        System.out.println("-".repeat(40));
        
        String[] names = new String[sortedResults.size()];
        double[] accuracies = new double[sortedResults.size()];
        
        for (int i = 0; i < sortedResults.size(); i++) {
            Map.Entry<String, Object> e = sortedResults.get(i);
            names[i] = e.getKey();
            accuracies[i] = (Double) e.getValue();
            System.out.printf("%-25s | %.4f%n", e.getKey(), (Double) e.getValue());
        }

        System.out.println("\nGenerating comparison charts...");
        ChartUtils.plotBarChart(names, accuracies, 
            "Classifier Accuracy Comparison", 
            "Classifier", "Accuracy",
            outputDir + "/classifier_comparison.png",
            1000, 600
        );

        System.out.println("\n" + "=".repeat(80));
        System.out.println("PCA DIMENSIONALITY REDUCTION");
        System.out.println("=".repeat(80));

        System.out.println("\nApplying PCA...");
        PCA pca = new PCA(2);
        Tensor X_pca = pca.fitTransform(X);
        
        System.out.println("Explained variance ratio: " + Arrays.toString(pca.getExplainedVarianceRatio()));
        System.out.println("Total explained variance: " + String.format("%.4f", 
            Arrays.stream(pca.getExplainedVarianceRatio()).sum()));

        double[] xData = X_pca.slice(1, 0, 1).getData();
        double[] yData = X_pca.slice(1, 1, 2).getData();
        double[] colors = y.getData();
        
        System.out.println("\nGenerating PCA scatter plot...");
        ChartUtils.plotScatter(xData, yData, "Iris Dataset - PCA Visualization",
            "PC1", "PC2", outputDir + "/pca_visualization.png");

        double[] pcaVar = pca.getExplainedVarianceRatio();
        String[] pcLabels = new String[pcaVar.length];
        for (int i = 0; i < pcLabels.length; i++) {
            pcLabels[i] = "PC" + (i + 1);
        }
        ChartUtils.plotBarChart(pcLabels, pcaVar,
            "PCA Explained Variance Ratio",
            "Principal Component", "Variance Ratio",
            outputDir + "/pca_variance.png");

        System.out.println("\n" + "=".repeat(80));
        System.out.println("REGRESSION EXAMPLE (BOSTON HOUSING)");
        System.out.println("=".repeat(80));

        System.out.println("\nLoading Boston Housing dataset...");
        BostonHousing boston = new BostonHousing();
        Tensor X_boston = boston.getFeatures();
        Tensor y_boston = boston.getTarget();

        System.out.println("Features: " + X_boston.shape().dim(1) + " features");
        System.out.println("Samples:  " + X_boston.shape().dim(0));

        Tensor[] bostonSplit = train_test_split.split(X_boston, y_boston, 0.3, 42);
        Tensor Xb_train = bostonSplit[0];
        Tensor Xb_test = bostonSplit[1];
        Tensor yb_train = bostonSplit[2];
        Tensor yb_test = bostonSplit[3];

        StandardScaler bostonScaler = new StandardScaler();
        Tensor Xb_train_scaled = bostonScaler.fitTransform(Xb_train);
        Tensor Xb_test_scaled = bostonScaler.transform(Xb_test);

        Map<String, Double> regResults = new LinkedHashMap<>();

        System.out.println("\nTraining regression models...");

        System.out.println("\n1. Linear Regression...");
        LinearRegression lr = new LinearRegression();
        lr.fit(Xb_train_scaled, yb_train);
        Tensor lr_pred = lr.predict(Xb_test_scaled);
        double lr_mse = MeanSquaredError.compute(yb_test, lr_pred);
        double lr_r2 = R2Score.compute(yb_test, lr_pred);
        regResults.put("Linear Regression", lr_r2);
        System.out.println("   MSE: " + String.format("%.4f", lr_mse));
        System.out.println("   R²:  " + String.format("%.4f", lr_r2));

        System.out.println("\n2. Ridge Regression...");
        Ridge ridge = new Ridge(1.0);
        ridge.fit(Xb_train_scaled, yb_train);
        Tensor ridge_pred = ridge.predict(Xb_test_scaled);
        double ridge_mse = MeanSquaredError.compute(yb_test, ridge_pred);
        double ridge_r2 = R2Score.compute(yb_test, ridge_pred);
        regResults.put("Ridge", ridge_r2);
        System.out.println("   MSE: " + String.format("%.4f", ridge_mse));
        System.out.println("   R²:  " + String.format("%.4f", ridge_r2));

        System.out.println("\n3. Random Forest Regressor...");
        RandomForestRegressor rf_reg = new RandomForestRegressor(50, 10, 2);
        rf_reg.fit(Xb_train, yb_train);
        Tensor rf_reg_pred = rf_reg.predict(Xb_test);
        double rf_mse = MeanSquaredError.compute(yb_test, rf_reg_pred);
        double rf_r2 = R2Score.compute(yb_test, rf_reg_pred);
        regResults.put("Random Forest", rf_r2);
        System.out.println("   MSE: " + String.format("%.4f", rf_mse));
        System.out.println("   R²:  " + String.format("%.4f", rf_r2));

        System.out.println("\n4. Decision Tree Regressor...");
        DecisionTreeRegressor dt_reg = new DecisionTreeRegressor(10, 2);
        dt_reg.fit(Xb_train, yb_train);
        Tensor dt_reg_pred = dt_reg.predict(Xb_test);
        double dt_mse = MeanSquaredError.compute(yb_test, dt_reg_pred);
        double dt_r2 = R2Score.compute(yb_test, dt_reg_pred);
        regResults.put("Decision Tree", dt_r2);
        System.out.println("   MSE: " + String.format("%.4f", dt_mse));
        System.out.println("   R²:  " + String.format("%.4f", dt_r2));

        System.out.println("\nGenerating regression comparison chart...");
        
        String[] regNames = new String[regResults.size()];
        double[] regScores = new double[regResults.size()];
        int idx = 0;
        for (Map.Entry<String, Double> e : regResults.entrySet()) {
            regNames[idx] = e.getKey();
            regScores[idx] = e.getValue();
            idx++;
        }
        
        ChartUtils.plotBarChart(regNames, regScores,
            "Regression Model R² Comparison",
            "Model", "R² Score",
            outputDir + "/regression_comparison.png");

        System.out.println("\nGenerating predictions vs actual chart...");
        double[] yb_actual = yb_test.getData();
        double[] lr_pred_arr = lr_pred.getData();
        
        ChartUtils.plotPredictions(yb_actual, lr_pred_arr,
            "Linear Regression: Predictions vs Actual",
            outputDir + "/regression_predictions.png",
            1000, 600
        );

        ChartUtils.plotResiduals(yb_actual, lr_pred_arr,
            "Linear Regression Residuals",
            outputDir + "/regression_residuals.png"
        );

        System.out.println("\n" + "=".repeat(80));
        System.out.println("CROSS-VALIDATION EXAMPLE");
        System.out.println("=".repeat(80));

        System.out.println("\nPerforming 5-fold cross-validation on Logistic Regression...");
        
        KFold kfold = new KFold(5, 42);
        List<Tensor[]> folds = kfold.split(X_train_scaled, y_train);
        
        double[] cvScores = new double[folds.size()];
        for (int i = 0; i < folds.size(); i++) {
            Tensor[] fold = folds.get(i);
            LogisticRegression cvModel = new LogisticRegression(0.01, 1000);
            cvModel.fit(fold[0], fold[2]);
            Tensor cvPred = cvModel.predict(fold[1]);
            cvScores[i] = Accuracy.compute(fold[3], cvPred);
            System.out.printf("   Fold %d: %.4f%n", i + 1, cvScores[i]);
        }
        
        double cvMean = Arrays.stream(cvScores).average().getAsDouble();
        double cvStd = Math.sqrt(Arrays.stream(cvScores).map(x -> Math.pow(x - cvMean, 2)).average().getAsDouble());
        
        System.out.printf("%n   Mean CV Accuracy: %.4f (±%.4f)%n", cvMean, cvStd);

        String[] foldLabels = new String[cvScores.length];
        for (int i = 0; i < foldLabels.length; i++) {
            foldLabels[i] = "Fold " + (i + 1);
        }
        ChartUtils.plotBarChart(foldLabels, cvScores,
            "5-Fold Cross-Validation Scores",
            "Fold", "Accuracy",
            outputDir + "/cross_validation.png"
        );

        System.out.println("\n" + "=".repeat(80));
        System.out.println("DEMONSTRATION COMPLETE");
        System.out.println("=".repeat(80));
        System.out.println("\nOutput files saved to: " + new File(outputDir).getAbsolutePath());
        System.out.println("  - classifier_comparison.png");
        System.out.println("  - pca_visualization.png");
        System.out.println("  - pca_variance.png");
        System.out.println("  - regression_comparison.png");
        System.out.println("  - regression_predictions.png");
        System.out.println("  - regression_residuals.png");
        System.out.println("  - cross_validation.png");
    }
}
