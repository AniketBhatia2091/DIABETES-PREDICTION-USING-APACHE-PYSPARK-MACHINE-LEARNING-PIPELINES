# Clinical Diabetes Prediction Portal using PySpark ML Pipelines

**A scalable machine learning pipeline and interactive clinical dashboard designed to predict diabetes using Apache PySpark & Streamlit.**

---

## 🚀 Project Overview
This project transforms a standard single-classifier model into a production-grade, distributed big data science solution to showcase scalable machine learning workflows, rigorous cross-validation grid search, and real-time medical diagnosis portals.

### 🛠️ Technology Stack
- **Data Engine**: Apache PySpark (Spark SQL & Spark ML)
- **Programming Language**: Python 3.9+
- **Inference App & UI**: Streamlit (Dashboard with custom typography and CSS)
- **Visualizations**: Plotly (EDA charts, metric comparison, risk gauge)
- **Environment Management**: Homebrew (for OpenJDK 17) & pip

---

## 🧬 System Architecture
The system consists of a data ingestion layer, a structured PySpark ML pipeline, and an interactive prediction front-end:

```
[Raw Clinical CSV] ─► [PySpark DataFrame] ─► [ML Pipeline (Imputer, Assembler, Scaler)]
                                                         │
[Streamlit App] ◄─── [Saved Model Pipeline] ◄────────────┴─── [CrossValidator Tuning]
```

### Key Stages inside the Spark ML Pipeline:
1. **Median Imputer**: Automatically replaces physiological `0`s (in Glucose, BP, Skin Fold, Insulin, and BMI) with the column-wise median values calculated from training sets.
2. **VectorAssembler**: Aggregates all numerical features into a single dense vector column.
3. **StandardScaler**: Performs Z-score standardization ($\mu=0, \sigma=1$) to prevent features with larger magnitudes from dominating the models.
4. **Tuned Classifier**: Leverages Grid Search to tune hyperparameters for **Logistic Regression**, **Random Forest**, **Gradient Boosted Trees (GBT)**, and **Linear Support Vector Classifiers (LinearSVC)**.

---

## 📁 Directory Structure
```
Diabetes-Prediction-main/
├── app/
│   └── main.py                     # Streamlit dashboard & live diagnosis interface
├── data/
│   ├── diabetes.csv                # Raw Pima Indians Diabetes dataset
│   └── processed/
│       ├── model_comparison.json   # Model performance benchmarking report
│       └── test_predictions.csv    # Test dataset predictions for visualization
├── models/
│   └── best_spark_model/           # Saved winner PySpark PipelineModel
├── notebooks/
│   └── diabetes_pyspark_pipeline.ipynb # Step-by-step presentation notebook
├── src/
│   ├── spark_pipeline.py           # Model training, hyperparameter tuning & evaluation script
│   └── predict.py                  # Standalone CLI prediction helper script
├── PROJECT_REPORT.md               # Formal capstone academic report
├── README.md                       # Setup and presentations guide
├── requirements.txt                # Python package list
└── setup_and_run.sh                # Automated setup, training, and launch script
```

---

## ⚡ Quick Start: Setup and Execution

To run the entire system in one command (including dependency installation, Java 17 detection, model training, and web dashboard launch), execute the automated shell script:

```bash
./setup_and_run.sh
```

### Manual Execution Steps:

#### 1. Configure Java 17 and Python Environment
PySpark requires JDK 8, 11, or 17. If you have Java 17 installed via Homebrew (`brew install openjdk@17`), configure your environment variables:
```bash
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
```

Install python packages:
```bash
python3 -m pip install -r requirements.txt
```

#### 2. Run the PySpark ML Training Pipeline
This script processes the raw CSV, performs K-Fold Cross-Validation tuning, benchmarks 4 classifiers, saves the winner model pipeline to `models/`, and writes the comparison metrics:
```bash
python3 src/spark_pipeline.py
```

#### 3. Test CLI Single Inference
Make a standalone prediction using the saved pipeline model:
```bash
python3 src/predict.py
```

#### 4. Launch the Web Application Portal
Open the interactive medical visualization dashboard in your browser:
```bash
streamlit run app/main.py
```

---

## 📊 Model Evaluation Summary

The following test metrics show GBT achieving the highest predictive capabilities:

| Model Pipeline | Accuracy | ROC-AUC | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | ~77.3% | ~0.835 | ~0.768 | ~0.770 | ~0.773 |
| **Random Forest** | ~78.6% | ~0.841 | ~0.781 | ~0.784 | ~0.786 |
| **Gradient Boosted Trees (GBT)** | **~79.9%** | **~0.852** | **~0.796** | **~0.797** | **~0.799** |
| **Linear Support Vector Machine**| ~77.9% | ~0.829 | ~0.771 | ~0.774 | ~0.779 |