#!/bin/bash

# Setup and Run script for Diabetes Prediction Project
echo "======================================================================"
echo "🩺 Starting Diabetes Prediction & Analytics System Setup"
echo "======================================================================"

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install Python 3.9+."
    exit 1
fi

# 2. Dynamically Resolve Java 17 (Homebrew openjdk@17 or system)
echo "🔍 Detecting Java 17 Installation..."
if [ -d "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
    export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
elif [ -d "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
    export JAVA_HOME="/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
else
    # Fallback to system java_home search for version 17
    DETECTED_JAVA=$(/usr/libexec/java_home -v 17 2>/dev/null)
    if [ ! -z "$DETECTED_JAVA" ]; then
        export JAVA_HOME="$DETECTED_JAVA"
    else
        echo "⚠️ Warning: JDK 17 not found explicitly. Falling back to default system JDK..."
        export JAVA_HOME=$(/usr/libexec/java_home 2>/dev/null)
    fi
fi

if [ -z "$JAVA_HOME" ]; then
    echo "❌ Error: Java is not installed. PySpark requires a JDK installed on the system."
    exit 1
fi

export PATH="$JAVA_HOME/bin:$PATH"
echo "✅ JAVA_HOME set to: $JAVA_HOME"
java -version

# 3. Install Python Dependencies
echo "📦 Installing Python package requirements..."
python3 -m pip install -r requirements.txt --user

# 4. Run PySpark Model Training
echo "🚀 Running PySpark Training & Hyperparameter Tuning Pipeline..."
python3 src/spark_pipeline.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Training pipeline failed. Check Java version compatibility or spark dependencies."
    exit 1
fi

# 5. Launch Streamlit Web App
echo "💻 Launching the clinical diagnosis web portal..."
python3 -m streamlit run app/main.py
