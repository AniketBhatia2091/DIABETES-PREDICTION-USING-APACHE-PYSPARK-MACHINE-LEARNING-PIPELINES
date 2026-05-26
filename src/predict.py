import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml import PipelineModel

def predict_single(input_data):
    """
    Predicts diabetes for a single patient record using the saved PySpark ML Pipeline.
    
    Parameters:
        input_data (dict): Dictionary with keys:
            - Pregnancies (int)
            - Glucose (float)
            - BloodPressure (float)
            - SkinThickness (float)
            - Insulin (float)
            - BMI (float)
            - DiabetesPedigreeFunction (float)
            - Age (int)
            
    Returns:
        prediction (int): 0 (Non-Diabetic) or 1 (Diabetic)
        probability (float): Risk probability (0.0 to 1.0)
    """
    # 1. Initialize temporary Spark Session (if not running)
    spark = SparkSession.builder \
        .appName("DiabetesPredictionInference") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # 2. Check if model exists
    model_path = "models/best_spark_model"
    if not os.path.exists(model_path):
        print(f"Error: Trained model pipeline not found at '{model_path}'. Please run training first via 'python src/spark_pipeline.py'.")
        sys.exit(1)
        
    # 3. Load model pipeline
    model = PipelineModel.load(model_path)
    
    # 4. Construct Schema
    schema = StructType([
        StructField("Pregnancies", IntegerType(), True),
        StructField("Glucose", DoubleType(), True),
        StructField("BloodPressure", DoubleType(), True),
        StructField("SkinThickness", DoubleType(), True),
        StructField("Insulin", DoubleType(), True),
        StructField("BMI", DoubleType(), True),
        StructField("DiabetesPedigreeFunction", DoubleType(), True),
        StructField("Age", IntegerType(), True)
    ])
    
    # Replace zeros with None for the columns that are imputed inside the pipeline
    # The pipeline's Imputer expects null values, not 0s!
    impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    cleaned_data = {}
    for col, val in input_data.items():
        if col in impute_cols and val == 0:
            cleaned_data[col] = None
        else:
            cleaned_data[col] = val
            
    # 5. Create PySpark DataFrame
    row_df = spark.createDataFrame([cleaned_data], schema=schema)
    
    # 6. Apply pipeline transformation
    transformed_df = model.transform(row_df)
    
    # 7. Extract prediction and probability
    # Fetch results
    results = transformed_df.select("prediction", *["probability" if "probability" in transformed_df.columns else "prediction"]).first()
    
    prediction = int(results["prediction"])
    
    if "probability" in transformed_df.columns:
        # probability is a dense vector [prob_0, prob_1]
        probability = float(results["probability"][1])
    else:
        probability = float(prediction)
        
    return prediction, probability

if __name__ == "__main__":
    # Standard sample case (diabetic case)
    sample_input = {
        "Pregnancies": 6,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 0.0,  # Will be imputed
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    
    print("Testing PySpark Model Pipeline Inference...")
    print("Input Patient Record:")
    for k, v in sample_input.items():
        print(f"  {k}: {v}")
        
    pred, prob = predict_single(sample_input)
    print("\n--- Diagnostic Prediction Results ---")
    print(f"Outcome Label: {pred} ({'Diabetic' if pred == 1 else 'Non-Diabetic'})")
    print(f"Diabetes Risk Probability: {prob * 100:.2f}%")
