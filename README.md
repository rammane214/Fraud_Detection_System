# 💳 Fraud Detection System

A machine learning-based system for detecting fraudulent credit card transactions using XGBoost classifier with real-time web interface.

## 🚀 Quick Start

**Run the Web App:**
```bash
streamlit run app/streamlit_app.py
```

Access the dashboard at `http://localhost:8501`

## 📊 Project Overview

- **Algorithm**: XGBoost Classifier
- **Features**: 31 features (Time, Amount, V1-V28 PCA components, Hour)
- **Dataset**: 284,807 transactions with 0.17% fraud rate
- **Accuracy**: High precision on imbalanced fraud detection
- **Interface**: Streamlit web dashboard for real-time predictions

## 📁 Project Structure

```
├── app/
│   └── streamlit_app.py      # Web interface for predictions
├── data/
│   ├── creditcard.csv        # Original dataset
│   ├── X_train.csv          # Training features (31 features)
│   ├── X_test.csv           # Test features
│   ├── y_train.csv          # Training labels
│   └── y_test.csv           # Test labels
├── models/
│   ├── xgboost.pkl          # Trained XGBoost model
│   ├── random_forest.pkl    # Alternative Random Forest model
│   └── scaler.pkl           # Feature scaler
├── notebooks/               # Jupyter notebooks for EDA and training
├── api/                     # API endpoints (future)
├── reports/                 # Analysis and documentation
└── assets/                  # Images and diagrams
```

## 🔧 Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n fraud_env python=3.10 -y
```

### 2. Activate Environment
```bash
conda activate fraud_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
# Web App
streamlit run app/streamlit_app.py

# Or for development
jupyter notebook
```

## 📈 Features Explained

The model uses **31 features** for fraud detection:

### Core Features
- **scaled_amount**: Transaction amount (normalized)
- **scaled_time**: Transaction timestamp (normalized)
- **Hour**: Hour of day (0-23) - critical for time-based fraud patterns

### PCA Features (V1-V28)
- 28 principal components from original transaction data
- Capture behavioral patterns: spending habits, location, device info
- Anonymized for privacy protection
- Range typically -3 to +3

### Feature Importance
- **Most critical**: V14, V12, V10, V17 (strong fraud correlation)
- **Amount**: Unusual transaction values
- **Hour**: Fraud peaks at 2AM-6AM
- **Combined patterns**: Multiple anomalies increase fraud probability

## 🤖 Model Information

### XGBoost Model (Primary)
- **File**: `models/xgboost.pkl`
- **Type**: Gradient boosting classifier
- **Training data**: 226,980 samples
- **Features**: 31 scaled features
- **Advantages**: Handles imbalanced data, high accuracy

### Random Forest Model (Alternative)
- **File**: `models/random_forest.pkl`
- **Type**: Ensemble decision trees
- **Use case**: Comparison and backup

### Scaler
- **File**: `models/scaler.pkl`
- **Purpose**: Normalizes input features
- **Applied to**: All 31 features before prediction

## 🎯 How It Works

1. **Input**: User enters transaction details (Amount, Time, Hour, V1-V28)
2. **Preprocessing**: Features are scaled using the saved scaler
3. **Prediction**: XGBoost model calculates fraud probability (0-1)
4. **Output**: Risk level (LOW/MEDIUM/HIGH) and decision (ALLOW/BLOCK)

## 🔍 Troubleshooting

### Feature Shape Mismatch
**Error**: "Feature shape mismatch, expected: 31, got 30"
**Solution**: Ensure all 31 features are provided (including Hour)

### Model Loading Issues
**Error**: Cannot load model files
**Solution**: Verify models exist in `models/` directory

### Dependency Conflicts
**Error**: Package version conflicts
**Solution**: Create fresh conda environment with Python 3.10

## 📝 Recent Changes

- **Fixed**: Added missing `Hour` feature to web app input
- **Updated**: Feature array now includes all 31 features as expected by model
- **Improved**: Error handling in prediction pipeline

## 📚 Data Files

- **creditcard.csv**: Original Kaggle dataset (284,807 transactions)
- **X_train.csv**: Training features with 31 columns
- **X_test.csv**: Test features for evaluation
- **y_train.csv/y_test.csv**: Binary labels (0=legitimate, 1=fraud)

## 🛠️ Development

### For Model Training
```bash
jupyter notebook
# Use notebooks/ for exploration and model training
```

### For API Development
```bash
# Implement endpoints in api/ directory
# Future enhancement for REST API
```

## 📊 Performance Metrics

- **Dataset**: Highly imbalanced (0.17% fraud rate)
- **Challenge**: Detecting rare fraud patterns
- **Approach**: SMOTE for handling class imbalance
- **Evaluation**: Precision, Recall, F1-score (not just accuracy)

## 📄 License

This project uses the Kaggle Credit Card Fraud Detection dataset for educational purposes.
