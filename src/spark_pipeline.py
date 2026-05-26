import os
import json
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, LinearSVC, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def create_spark_session():
    """Initializes Spark Session with local optimization configuration."""
    print("Initializing PySpark Session...")
    spark = SparkSession.builder \
        .appName("DiabetesPredictionPipeline") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark Session established. Spark version: {spark.version}")
    return spark

def load_and_preprocess_data(spark, data_path):
    """Loads CSV and handles zero-values representing missing data."""
    print(f"Loading dataset from {data_path}...")
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # In Pima Indians dataset, 0 in these columns indicates missing medical records
    impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    print("Preprocessing data: Replacing 0 values with Null/None in medical features for imputation...")
    for col in impute_cols:
        df = df.withColumn(col, F.when(F.col(col) == 0, None).otherwise(F.col(col)))
        
    return df

def build_preprocessing_stages():
    """Defines Pipeline stages for Imputation, Vector Assembly, and Scaling."""
    impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    output_impute_cols = [f"{c}_imputed" for c in impute_cols]
    
    # 1. Imputer Stage (Median Strategy)
    imputer = Imputer(
        inputCols=impute_cols,
        outputCols=output_impute_cols,
        strategy="median"
    )
    
    # 2. Vector Assembler Stage
    # Use features that don't need imputation + imputed features
    feature_cols = [
        "Pregnancies", 
        "Glucose_imputed", 
        "BloodPressure_imputed", 
        "SkinThickness_imputed", 
        "Insulin_imputed", 
        "BMI_imputed", 
        "DiabetesPedigreeFunction", 
        "Age"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
    
    # 3. StandardScaler Stage (Z-score normalization)
    scaler = StandardScaler(
        inputCol="assembled_features", 
        outputCol="features", 
        withStd=True, 
        withMean=True
    )
    
    return imputer, assembler, scaler

def train_and_tune_models(train_df, test_df, preprocessing_stages):
    """Trains multiple classifiers using cross-validation and hyperparameter grids."""
    imputer, assembler, scaler = preprocessing_stages
    
    # Define models
    lr = LogisticRegression(labelCol="Outcome", featuresCol="features")
    rf = RandomForestClassifier(labelCol="Outcome", featuresCol="features", seed=42)
    gbt = GBTClassifier(labelCol="Outcome", featuresCol="features", seed=42)
    dt = DecisionTreeClassifier(labelCol="Outcome", featuresCol="features", seed=42)
    svc = LinearSVC(labelCol="Outcome", featuresCol="features")
    
    models = {
        "LogisticRegression": (lr, 
                               ParamGridBuilder()
                               .addGrid(lr.regParam, [0.01, 0.1, 0.5])
                               .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                               .build()),
        "DecisionTree": (dt,
                         ParamGridBuilder()
                         .addGrid(dt.maxDepth, [3, 5, 7])
                         .build()),
        "RandomForest": (rf, 
                         ParamGridBuilder()
                         .addGrid(rf.numTrees, [10, 30, 50])
                         .addGrid(rf.maxDepth, [5, 7, 9])
                         .build()),
        "GradientBoosting": (gbt, 
                             ParamGridBuilder()
                             .addGrid(gbt.maxIter, [10, 20, 30])
                             .addGrid(gbt.maxDepth, [3, 5])
                             .build()),
        "SupportVectorMachine": (svc, 
                                 ParamGridBuilder()
                                 .addGrid(svc.maxIter, [50, 100])
                                 .addGrid(svc.regParam, [0.01, 0.1])
                                 .build())
    }
    
    # Evaluators
    evaluator_roc = BinaryClassificationEvaluator(labelCol="Outcome", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName="f1")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction", metricName="weightedRecall")
    
    model_metrics = {}
    best_pipelines = {}
    best_score = 0.0
    best_model_name = ""
    
    for name, (classifier, paramGrid) in models.items():
        print(f"\n--- Tuning Hyperparameters for {name} ---")
        
        # Pipeline including preprocessing and the classifier
        pipeline = Pipeline(stages=[imputer, assembler, scaler, classifier])
        
        # Cross Validator (3-fold cross-validation)
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator_roc,
            numFolds=3,
            seed=42
        )
        
        print(f"Fitting CV Grid on training data...")
        cv_model = cv.fit(train_df)
        
        # Predict on test data
        predictions = cv_model.transform(test_df)
        
        # Compute metrics
        roc_auc = evaluator_roc.evaluate(predictions)
        accuracy = evaluator_acc.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)
        precision = evaluator_precision.evaluate(predictions)
        recall = evaluator_recall.evaluate(predictions)
        
        print(f"[{name}] Results on Test Set:")
        print(f"  - Accuracy:  {accuracy:.4f}")
        print(f"  - ROC-AUC:   {roc_auc:.4f}")
        print(f"  - F1-Score:  {f1:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        
        model_metrics[name] = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }
        
        # Track the best pipeline model using Test Set ROC-AUC score
        best_pipelines[name] = cv_model.bestModel
        if roc_auc > best_score:
            best_score = roc_auc
            best_model_name = name
            
    print(f"\nWinner Model: {best_model_name} with ROC-AUC = {best_score:.4f}")
    
    return best_pipelines, model_metrics, best_model_name

def save_artifacts(best_pipelines, metrics, best_model_name, test_df):
    """Saves models, comparison metrics JSON, and test predictions for Dashboard use."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Save the winner model pipeline
    best_model_path = "models/best_spark_model"
    print(f"Saving the best model ({best_model_name}) pipeline to {best_model_path}...")
    best_pipelines[best_model_name].write().overwrite().save(best_model_path)
    
    # Save comparison metrics JSON
    metrics_path = "data/processed/model_comparison.json"
    print(f"Saving comparison report to {metrics_path}...")
    
    # Extract feature coefficients from LogisticRegression if present
    if "LogisticRegression" in best_pipelines:
        lr_model = best_pipelines["LogisticRegression"].stages[-1]
        metrics["LogisticRegression_coefficients"] = {
            "coefficients": lr_model.coefficients.toArray().tolist(),
            "intercept": float(lr_model.intercept),
            "features": [
                "Pregnancies", 
                "Glucose", 
                "BloodPressure", 
                "SkinThickness", 
                "Insulin", 
                "BMI", 
                "DiabetesPedigreeFunction", 
                "Age"
            ]
        }
        
    metrics["meta"] = {
        "best_model_name": best_model_name,
        "best_roc_auc": metrics[best_model_name]["roc_auc"]
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save test prediction outcomes for the dashboard to generate ROC curve data points
    print("Generating prediction points on the test set for Plotly curves...")
    # Generate predictions using the best model
    best_model = best_pipelines[best_model_name]
    test_predictions = best_model.transform(test_df)
    
    # For SVC, rawPrediction has two dimensions but it's a vector; we'll extract probabilities or scores
    # Since GBT and RF are standard, let's extract raw predictions, probability, and outcomes to pandas
    pandas_preds = test_predictions.select(
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome", "prediction"
    ).toPandas()
    
    # If probability is present (RF, LR, GBT), save it too for ROC curves
    if "probability" in test_predictions.columns:
        prob_list = test_predictions.select("probability").rdd.map(lambda row: float(row.probability[1])).collect()
        pandas_preds["probability"] = prob_list
    else:
        # For SVC, map rawPrediction score to a pseudo-probability using sigmoid
        if "rawPrediction" in test_predictions.columns:
            score_list = test_predictions.select("rawPrediction").rdd.map(lambda row: float(row.rawPrediction[0])).collect()
            # Convert decision function score to pseudo probability
            pandas_preds["probability"] = 1.0 / (1.0 + np.exp(score_list))
        else:
            pandas_preds["probability"] = pandas_preds["prediction"]

    pandas_preds.to_csv("data/processed/test_predictions.csv", index=False)
    print("Artifacts saved successfully!")

def main():
    spark = create_spark_session()
    
    try:
        data_path = "data/diabetes.csv"
        df = load_and_preprocess_data(spark, data_path)
        
        # Split features stratifying by random split
        print("Splitting dataset into 80% Train and 20% Test sets...")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        
        # Cache dfs for fast training iterations
        train_df.cache()
        test_df.cache()
        
        preprocessing_stages = build_preprocessing_stages()
        
        best_pipelines, metrics, best_model_name = train_and_tune_models(
            train_df, test_df, preprocessing_stages
        )
        
        save_artifacts(best_pipelines, metrics, best_model_name, test_df)
        
    finally:
        print("Stopping Spark Session...")
        spark.stop()
        print("Session stopped.")

if __name__ == "__main__":
    main()
