import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import time

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Diabetes Predictor & Analytics Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS with Google Fonts (Poppins) and Glassmorphism design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef2f5;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .reportview-container {
        background: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions & Caching -----------------

@st.cache_resource
def get_spark_and_model():
    """Initializes Spark once and caches it to make web app inference instantaneous."""
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    
    spark = SparkSession.builder \
        .appName("StreamlitDiabetesApp") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    model_path = "models/best_spark_model"
    model = None
    if os.path.exists(model_path):
        model = PipelineModel.load(model_path)
        
    return spark, model

@st.cache_data
def load_historical_data():
    """Loads dataset for EDA purposes."""
    csv_path = "data/diabetes.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_comparison_metrics():
    """Loads metrics JSON generated during PySpark pipeline training."""
    json_path = "data/processed/model_comparison.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_test_predictions():
    """Loads test predictions CSV containing predicted probabilities."""
    pred_path = "data/processed/test_predictions.csv"
    if os.path.exists(pred_path):
        return pd.read_csv(pred_path)
    return None

# Load cached items
df = load_historical_data()
metrics_data = load_comparison_metrics()
test_preds = load_test_predictions()

# ----------------- Page Layout & Navigation -----------------

st.sidebar.markdown(
    '<div style="text-align: center; padding-bottom: 10px;">'
    '<h2 style="color: #1e3c72;">🩺 Diabetes ML Portal</h2>'
    '<p style="color: #64748b; font-size: 0.85rem;">Clinical Analytics Platform<br>Powered by Apache PySpark ML</p>'
    '</div>', 
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go To",
    ["🏢 Project Overview & EDA", "📊 Spark Model Performance", "🩺 Real-Time Patient Diagnostic", "📡 Live EHR Patient Stream"]
)

# ----------------- 1. Project Overview & EDA Page -----------------

if page == "🏢 Project Overview & EDA":
    st.markdown('<div class="main-header"><h1>Diabetes Prediction & Diagnostic System</h1><p>A scalable big data analytics pipeline using PySpark ML Pipeline & Cross-Validation</p></div>', unsafe_allow_html=True)
    
    st.subheader("📊 System Highlights & Clinical Overview")
    
    if df is not None:
        total_patients = len(df)
        diabetic_count = int(df["Outcome"].sum())
        non_diabetic_count = total_patients - diabetic_count
        diabetic_pct = (diabetic_count / total_patients) * 100
        avg_glucose = df["Glucose"].mean()
        avg_bmi = df["BMI"].mean()
        
        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_patients}</div><div class="metric-label">Total Patients</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{diabetic_pct:.1f}%</div><div class="metric-label">Diabetes Prevalence</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_glucose:.1f}</div><div class="metric-label">Avg Glucose (mg/dL)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_bmi:.1f}</div><div class="metric-label">Average BMI</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Split layout for charts
        chart_col1, chart_col2 = st.columns([1.1, 0.9])
        
        with chart_col1:
            st.markdown("### 📈 Clinical Feature Distributions")
            feature_to_plot = st.selectbox(
                "Select Feature to Visualize",
                ["Glucose", "BMI", "Age", "BloodPressure", "Insulin", "SkinThickness", "Pregnancies", "DiabetesPedigreeFunction"]
            )
            
            fig = px.histogram(
                df, 
                x=feature_to_plot, 
                color="Outcome", 
                barmode="overlay",
                color_discrete_map={0: "#0ea5e9", 1: "#ef4444"},
                labels={"Outcome": "Diagnosis"},
                title=f"Distribution of {feature_to_plot} segmented by Outcome (0: Healthy, 1: Diabetic)"
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with chart_col2:
            st.markdown("### 🧬 Clinical Feature Correlation Heatmap")
            # Calculate correlation matrix
            corr = df.corr()
            fig_corr = px.imshow(
                corr, 
                text_auto=".2f", 
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Pearson Correlation Heatmap"
            )
            fig_corr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        st.markdown("""
        ---
        ### 📌 Key Clinical Discoveries in the Dataset:
        1. **Glucose Levels**: Patients diagnosed with diabetes show a significantly higher distribution in glucose levels, heavily centered above 140 mg/dL.
        2. **Body Mass Index (BMI)**: A clear correlation exists between higher BMI and positive diabetes diagnosis, indicating obesity is a primary risk factor.
        3. **Age & Pregnancies**: Multi-pregnancy patients and older cohorts display a higher incidence rate, reflecting gestational risk and aging factors.
        """)
        
    else:
        st.warning("⚠️ Raw dataset `data/diabetes.csv` not found. Please run the setup script to download the dataset.")

# ----------------- 2. Model Performance Page -----------------

elif page == "📊 Spark Model Performance":
    st.markdown('<div class="main-header"><h1>PySpark Pipeline Benchmarking</h1><p>Comparative analysis of Spark ML models tuned using K-Fold Cross-Validation</p></div>', unsafe_allow_html=True)
    
    if metrics_data is not None:
        best_model = metrics_data["meta"]["best_model_name"]
        best_roc = metrics_data["meta"]["best_roc_auc"]
        
        st.markdown(f"""
        <div style="background-color: #f0fdf4; border-left: 5px solid #22c55e; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
            <h3 style="color: #15803d; margin-top:0;">🏆 Current Champion Model: {best_model}</h3>
            <p style="color: #166534; font-size:1.05rem; margin:0;">
                The PySpark pipeline has evaluated Logistic Regression, Random Forest, GBT, and SVM. 
                The best performing model on the test set is <b>{best_model}</b> with an Area Under the ROC Curve (ROC-AUC) of <b>{best_roc:.4f}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metric comparison plot
        models_list = [name for name in metrics_data.keys() if name not in ["meta", "LogisticRegression_coefficients"]]
        
        comparison_dict = {
            "Model": [],
            "Metric": [],
            "Score": []
        }
        
        for model_name in models_list:
            for metric_name in ["accuracy", "roc_auc", "f1_score", "precision", "recall"]:
                comparison_dict["Model"].append(model_name)
                comparison_dict["Metric"].append(metric_name.upper().replace("_", "-"))
                comparison_dict["Score"].append(metrics_data[model_name][metric_name])
                
        metrics_df = pd.DataFrame(comparison_dict)
        
        fig_metrics = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_sequence=["#1e3c72", "#0ea5e9", "#10b981", "#f59e0b"],
            title="Comparison of Performance Metrics across Spark Models",
            labels={"Score": "Score (0.0 to 1.0)"}
        )
        fig_metrics.update_layout(
            yaxis_range=[0.5, 1.0],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Side-by-side: Metric table & ROC curves
        col1, col2 = st.columns([1, 1.1])
        
        with col1:
            st.markdown("### 📋 Model Metrics Comparison Table")
            table_data = []
            for m in models_list:
                row = {"Model": m}
                for metric in ["accuracy", "roc_auc", "f1_score", "precision", "recall"]:
                    row[metric.upper().replace("_", "-")] = f"{metrics_data[m][metric]:.4f}"
                table_data.append(row)
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            
            # Show Feature importance coefficients if available
            if "LogisticRegression_coefficients" in metrics_data:
                st.markdown("### 🔬 Spark Global Model Weights")
                coef_info = metrics_data["LogisticRegression_coefficients"]
                coefs = coef_info["coefficients"]
                features = coef_info["features"]
                
                # Make dataframe
                feat_imp_df = pd.DataFrame({
                    "Feature": features,
                    "Coefficient (Weight)": coefs
                }).sort_values(by="Coefficient (Weight)", key=abs, ascending=True)
                
                # Plot horizontal bar chart
                fig_coef = px.bar(
                    feat_imp_df,
                    x="Coefficient (Weight)",
                    y="Feature",
                    orientation="h",
                    color="Coefficient (Weight)",
                    color_continuous_scale="RdBu",
                    title="Logistic Regression Feature Coefficients (Scaled Features)"
                )
                fig_coef.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_coef, use_container_width=True)
                
        with col2:
            st.markdown("### 📈 Interactive ROC Curve Visualization")
            if test_preds is not None and "probability" in test_preds.columns:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(test_preds["Outcome"], test_preds["probability"])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = px.area(
                    x=fpr, y=tpr,
                    title=f"Receiver Operating Characteristic (ROC) Curve (AUC = {roc_auc:.4f})",
                    labels=dict(x="False Positive Rate (1 - Specificity)", y="True Positive Rate (Sensitivity)"),
                )
                fig_roc.add_shape(
                    type='line', line=dict(dash='dash', color='gray'),
                    x0=0, x1=1, y0=0, y1=1
                )
                fig_roc.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Test set predictions not available. Run pipeline training to generate ROC curve data points.")
            
            st.markdown("""
            The training process executes inside a unified Spark Pipeline:
            
            ```
            [Raw Input DataFrame]
                      │
                      ▼
            [1. Imputer (Median)] ──► Replaces 0s in Glucose, BP, Skin, Insulin, BMI
                      │
                      ▼
            [2. VectorAssembler]  ──► Merges all 8 numerical inputs into a single Vector
                      │
                      ▼
            [3. StandardScaler]   ──► Centers and scales features (Z-Score scaling)
                      │
                      ▼
            [4. Spark Classifier] ──► Fits ML models (LogisticReg, RandomForest, GBT, SVM)
            ```
            """)
            
    else:
        st.warning("⚠️ Benchmarking data not found. Run training: `python src/spark_pipeline.py` to generate metrics.")

# ----------------- 3. Real-Time Patient Diagnostic Page -----------------

elif page == "🩺 Real-Time Patient Diagnostic":
    st.markdown('<div class="main-header"><h1>Patient Diagnostic & Inference Portal</h1><p>Generate instant medical diagnoses from our trained PySpark pipeline model</p></div>', unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists("models/best_spark_model"):
        st.error("🚨 Trained Spark Model Pipeline not found! Run training first: `python src/spark_pipeline.py`.")
    else:
        # Load Spark & Model (cached resource)
        with st.spinner("Initializing PySpark Session and loading Saved Pipeline Model..."):
            spark_sess, model = get_spark_and_model()
            
        st.success("🤖 PySpark Session loaded and Spark ML pipeline active!")
        
        # User input form
        st.markdown("### 📝 Input Patient Clinical Measurements")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pregnancies = st.slider("Pregnancies (Count)", min_value=0, max_value=20, value=2, step=1)
            glucose = st.number_input("Plasma Glucose (mg/dL)", min_value=0, max_value=300, value=120)
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            
        with col2:
            skin_thickness = st.slider("Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20)
            insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=80.0, value=28.5, step=0.1)
            
        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.45, step=0.01)
            age = st.slider("Age (Years)", min_value=1, max_value=100, value=30)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Diagnose button
        if st.button("🔍 Run Diagnostic Analysis", type="primary", use_container_width=True):
            # Construct patient dictionary
            patient_record = {
                "Pregnancies": int(pregnancies),
                "Glucose": float(glucose),
                "BloodPressure": float(blood_pressure),
                "SkinThickness": float(skin_thickness),
                "Insulin": float(insulin),
                "BMI": float(bmi),
                "DiabetesPedigreeFunction": float(dpf),
                "Age": int(age)
            }
            
            with st.spinner("Processing through PySpark ML Pipeline..."):
                from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
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
                
                impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
                cleaned_record = {}
                for col_name, val in patient_record.items():
                    if col_name in impute_cols and val == 0:
                        cleaned_record[col_name] = None
                    else:
                        cleaned_record[col_name] = val
                
                patient_df = spark_sess.createDataFrame([cleaned_record], schema=schema)
                predictions = model.transform(patient_df)
                results = predictions.select("prediction", *["probability" if "probability" in predictions.columns else "prediction"]).first()
                
                pred_label = int(results["prediction"])
                
                if "probability" in predictions.columns:
                    prob_risk = float(results["probability"][1])
                else:
                    prob_risk = 1.0 if pred_label == 1 else 0.0
            
            st.markdown("---")
            st.markdown("## 📋 Diagnostic Evaluation Report")
            
            res_col1, res_col2 = st.columns([1, 1.2])
            
            # Formulate clinical guidance
            guidelines = []
            if glucose > 140:
                guidelines.append("- **Glucose Control**: Plasma glucose is high (>140 mg/dL). Reduce simple carbohydrate and sugar intake. Daily glycemic monitoring is advised.")
            if bmi > 25:
                guidelines.append(f"- **Weight Optimization**: BMI is in the overweight/obese range ({bmi:.1f}). Aiming for a 5-10% reduction in weight through regular cardio and caloric control can reverse risk factor trends.")
            if blood_pressure > 80:
                guidelines.append("- **Hypertension Management**: Diastolic BP is elevated (>80 mm Hg). Adopt a low-sodium DASH diet and monitor cardiovascular metrics.")
            if age > 45:
                guidelines.append("- **Age factor**: Patient is in the >45 age bracket. Annual screening is recommended due to increased metabolic resistance.")
            if not guidelines:
                guidelines.append("- Excellent indicators! Keep up the balanced nutritional choices and active exercise routine.")

            with res_col1:
                if pred_label == 1:
                    st.markdown(f"""
                    <div style="background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 1.5rem; border-radius: 8px;">
                        <h2 style="color: #b91c1c; margin-top:0;">🛑 Diagnosis: POSITIVE</h2>
                        <p style="color: #991b1b; font-size:1.1rem; margin:0;">
                            The diagnostic pipeline indicates a **High Risk** of diabetes ({prob_risk*100:.1f}% score). Clinical intervention is advised.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f0fdf4; border-left: 5px solid #22c55e; padding: 1.5rem; border-radius: 8px;">
                        <h2 style="color: #15803d; margin-top:0;">✅ Diagnosis: NEGATIVE</h2>
                        <p style="color: #166534; font-size:1.1rem; margin:0;">
                            The diagnostic pipeline indicates a **Low Risk** of diabetes ({prob_risk*100:.1f}% score). Regular checkups are advised.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 💡 Tailored Clinical Guidance:")
                for guide in guidelines:
                    st.markdown(guide)
                
                # DOWNLOAD REPORT BUTTON
                report_content = f"""======================================================
🩺 CLINICAL METABOLIC DIAGNOSTIC REPORT
Generated by: PySpark ML Pipeline Diagnostic Engine
======================================================
PATIENT MEASUREMENTS:
- Pregnancies: {pregnancies}
- Plasma Glucose: {glucose} mg/dL
- Diastolic Blood Pressure: {blood_pressure} mm Hg
- Triceps Skin Fold: {skin_thickness} mm
- 2-Hour Serum Insulin: {insulin} mu U/ml
- Body Mass Index (BMI): {bmi} kg/m^2
- Diabetes Pedigree Function: {dpf}
- Age: {age} Years

------------------------------------------------------
DIAGNOSTIC OUTCOME:
- Predictive Diagnostic: {"POSITIVE" if pred_label == 1 else "NEGATIVE"}
- Confidence Score: {prob_risk * 100:.2f}%
- Target Imputations applied: Medians derived via training Spark context.

------------------------------------------------------
CLINICAL ADVICE & ACTIONABLE INSIGHTS:
{chr(10).join(guidelines)}
======================================================
"""
                st.download_button(
                    label="📥 Download Clinical Diagnostic Report",
                    data=report_content,
                    file_name=f"patient_report_{int(time.time())}.txt",
                    mime="text/plain"
                )
                    
            with res_col2:
                # Plotly Gauge Chart for Risk Probability
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_risk * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Calculated Diabetes Risk Probability (%)", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#1e3c72"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 35], 'color': '#d1fae5'},
                            {'range': [35, 70], 'color': '#fef3c7'},
                            {'range': [70, 100], 'color': '#fee2e2'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=30, b=10, l=10, r=10),
                    height=300
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)

# ----------------- 4. Live EHR Patient Stream Page (EXTRA!) -----------------

elif page == "📡 Live EHR Patient Stream":
    st.markdown('<div class="main-header"><h1>📡 Real-Time EHR Patient Stream Monitor</h1><p>Simulating clinical patient arrivals and running live classifications using Structured Spark Pipelines</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    This simulator models real-time clinical patient vital signs streaming from hospital emergency units or clinical EHR feeds. 
    Incoming patient measurements are processed instantly by the serialized PySpark ML Pipeline to stratify risk.
    """)
    
    # Check if model exists
    if not os.path.exists("models/best_spark_model"):
        st.error("🚨 Trained Spark Model Pipeline not found! Run training first.")
    else:
        # Load Spark & Model (cached resource)
        spark_sess, model = get_spark_and_model()
        
        # Stream control buttons
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            run_stream = st.checkbox("📡 Connect to EHR Feed & Run Stream", value=False)
        
        # Display structures
        metric_area = st.empty()
        chart_area = st.empty()
        table_area = st.empty()
        
        # We store streaming history in session_state to avoid losing it on rerun
        if 'stream_history' not in st.session_state:
            st.session_state['stream_history'] = []
            
        if run_stream:
            # Struct for PySpark DataFrame schema
            from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
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
            
            impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            
            while run_stream:
                # 1. Generate realistic patient vitals with random noise
                # Draw healthy or high-risk patients
                is_high_risk = np.random.choice([True, False], p=[0.35, 0.65])
                
                if is_high_risk:
                    glucose_val = int(np.random.normal(160, 25))
                    bmi_val = round(float(np.random.normal(36.0, 5.0)), 1)
                    age_val = int(np.random.randint(35, 65))
                    preg_val = int(np.random.randint(2, 9))
                else:
                    glucose_val = int(np.random.normal(105, 15))
                    bmi_val = round(float(np.random.normal(25.5, 3.5)), 1)
                    age_val = int(np.random.randint(21, 40))
                    preg_val = int(np.random.randint(0, 4))
                    
                bp_val = int(np.random.normal(72, 8))
                skin_val = int(np.random.normal(20, 10))
                insulin_val = int(np.random.choice([0, int(np.random.normal(120, 40))])) # Mock missing values
                dpf_val = round(float(np.random.exponential(0.4)), 3)
                
                # Clip values to realistic medical bounds
                glucose_val = max(40, min(250, glucose_val))
                bmi_val = max(15.0, min(60.0, bmi_val))
                bp_val = max(40, min(140, bp_val))
                skin_val = max(0, min(60, skin_val))
                insulin_val = max(0, min(600, insulin_val))
                dpf_val = max(0.08, min(2.5, dpf_val))
                
                # Make row record
                raw_record = {
                    "Pregnancies": preg_val,
                    "Glucose": float(glucose_val),
                    "BloodPressure": float(bp_val),
                    "SkinThickness": float(skin_val),
                    "Insulin": float(insulin_val),
                    "BMI": float(bmi_val),
                    "DiabetesPedigreeFunction": float(dpf_val),
                    "Age": age_val
                }
                
                # Preprocess zero imputation
                cleaned_rec = {}
                for col_name, val in raw_record.items():
                    if col_name in impute_cols and val == 0:
                        cleaned_rec[col_name] = None
                    else:
                        cleaned_rec[col_name] = val
                        
                # Create Spark DataFrame and run prediction
                patient_spark_df = spark_sess.createDataFrame([cleaned_rec], schema=schema)
                transformed_preds = model.transform(patient_spark_df)
                result_first = transformed_preds.select("prediction", *["probability" if "probability" in transformed_preds.columns else "prediction"]).first()
                
                lbl = int(result_first["prediction"])
                prob = float(result_first["probability"][1]) if "probability" in transformed_preds.columns else float(lbl)
                
                # Save to streaming history list
                stream_record = {
                    "Patient ID": f"PT-{np.random.randint(1000, 9999)}",
                    "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Glucose": glucose_val,
                    "Blood Pressure": bp_val,
                    "BMI": bmi_val,
                    "Age": age_val,
                    "Risk Score": round(prob * 100, 1),
                    "Diagnostic": "POSITIVE (Diabetic)" if lbl == 1 else "NEGATIVE (Healthy)"
                }
                
                # Prepend to display latest first
                st.session_state['stream_history'].insert(0, stream_record)
                
                # Limit history to 20
                if len(st.session_state['stream_history']) > 20:
                    st.session_state['stream_history'].pop()
                    
                # 2. Render Live Metrics in Metric Card Area
                history_df = pd.DataFrame(st.session_state['stream_history'])
                total_streamed = len(history_df)
                positives_streamed = len(history_df[history_df["Diagnostic"] == "POSITIVE (Diabetic)"])
                pct_positives = (positives_streamed / total_streamed) * 100 if total_streamed > 0 else 0.0
                
                with metric_area.container():
                    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                    with col_kpi1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_streamed}</div><div class="metric-label">Vitals Transmitted</div></div>', unsafe_allow_html=True)
                    with col_kpi2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{pct_positives:.1f}%</div><div class="metric-label">Stream Diabetes Rate</div></div>', unsafe_allow_html=True)
                    with col_kpi3:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{stream_record["Risk Score"]}%</div><div class="metric-label">Latest Patient Risk</div></div>', unsafe_allow_html=True)
                
                # 3. Render Risk Trend Line Chart
                with chart_area.container():
                    # Reverse history for logical timeline left-to-right plotting
                    timeline_df = history_df.iloc[::-1].copy()
                    fig_trend = px.line(
                        timeline_df,
                        x="Time",
                        y="Risk Score",
                        markers=True,
                        line_shape="linear",
                        title="Live Patient Risk Stream Timeline",
                        labels={"Risk Score": "Risk Score (%)", "Time": "Feed Time"}
                    )
                    fig_trend.update_layout(
                        yaxis_range=[0, 100],
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                
                # 4. Render Table
                with table_area.container():
                    st.markdown("### 📋 Live Ingested Patients Feed")
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
                    
                # Pause for 2 seconds to simulate network ingestion delay
                time.sleep(2.0)
                
        else:
            if st.session_state['stream_history']:
                st.info("EHR connection paused. Checking latest feed state.")
                history_df = pd.DataFrame(st.session_state['stream_history'])
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("💡 Connect to the EHR feed using the checkbox above to begin streaming data.")
