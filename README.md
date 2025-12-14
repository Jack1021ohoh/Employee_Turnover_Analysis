# Employee Turnover Analysis

A comprehensive machine learning project for predicting employee turnover using multiple classification algorithms and SHAP-based interpretability analysis.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Key Findings](#key-findings)

## Overview

This project analyzes and predicts employee turnover using a dataset of 4,653 employees across various features including education, city, payment tier, age, gender, and experience. The analysis employs statistical testing, exploratory data analysis (EDA), feature engineering, and multiple machine learning models to identify key factors influencing employee departure decisions.

## Dataset

**Source:** [Kaggle - Employee Dataset](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset)

**Dataset Characteristics:**
- Total Records: 4,653 employees
- Features: 9 columns (8 features + 1 target)
- Target Variable: `LeaveOrNot` (Binary classification: 0 = Stay, 1 = Leave)

**Features:**
- `Education`: Educational qualifications (Bachelors, Masters, PHD)
- `JoiningYear`: Year the employee joined the company
- `City`: Employee location (Bangalore, Pune, New Delhi)
- `PaymentTier`: Salary tier categorization (1-3, lower number = higher tier)
- `Age`: Employee age
- `Gender`: Male/Female
- `EverBenched`: Whether employee was ever temporarily without assigned work
- `ExperienceInCurrentDomain`: Years of experience in current field

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Univariate analysis using count plots and box plots
- Bivariate analysis examining relationships with target variable
- Statistical hypothesis testing (Chi-square for categorical, t-test for numerical features)
- All features showed statistical significance (p < 0.05)

### 2. Data Preprocessing
- **Train-Test Split:** 80-20 stratified split (3,722 training / 931 testing samples)
- **Categorical Encoding:** Leave-One-Out Target Encoding with sigma=0.05
- **Numerical Scaling:** StandardScaler for feature normalization
- **Class Imbalance Handling:** BorderlineSMOTE for oversampling minority class

### 3. Model Training & Evaluation
Six classification algorithms were trained and compared:
- Support Vector Machine (SVM) with multiple kernels
- Multi-Layer Perceptron (MLP)
- XGBoost
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

### 4. Model Interpretability
- SHAP (SHapley Additive exPlanations) analysis for feature importance
- Feature importance ranking across different models

## Models Implemented

| Model | Configuration | Best Parameters |
|-------|--------------|-----------------|
| **SVM** | RBF, Linear, Poly, Sigmoid kernels | RBF kernel performed best |
| **MLP** | Early stopping, momentum=0.99 | max_iter=200-600 |
| **XGBoost** | Default configuration with random_state | - |
| **Logistic Regression** | L2 regularization | - |
| **Decision Tree** | GridSearchCV optimization | max_depth=8, min_samples_leaf=5 |
| **KNN** | GridSearchCV optimization | n_neighbors=9 |

## Results

### Model Performance Comparison (with BorderlineSMOTE)

| Model | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|----------|-----|
| **XGBoost** | 0.785 | 0.706 | 0.743 | **0.865** |
| **SVM (RBF)** | 0.698 | 0.759 | 0.728 | 0.847 |
| **MLP** | 0.829 | 0.634 | 0.719 | 0.838 |

**Best Model:** XGBoost achieved the highest AUC score of 0.865 and balanced performance across all metrics.

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost category-encoders shap scipy
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
```bash
git clone <repository-url>
cd Employee_Turnover_Analysis
```

2. Open Jupyter notebooks:
```bash
jupyter notebook
```

3. Run the notebooks in sequence:
   - [EDA.ipynb](EDA.ipynb) - Exploratory Data Analysis
   - [BI_project.ipynb](BI_project.ipynb) - Model training and evaluation

## Project Structure

```
Employee_Turnover_Analysis/
├── Employee.csv              # Dataset
├── EDA.ipynb                 # Exploratory Data Analysis
├── BI_project.ipynb          # Model training and evaluation
├── picture/                  # Generated visualizations
└── README.md                 # Project documentation
```

## Technologies

**Core Libraries:**
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Imbalanced Data:** imbalanced-learn
- **Feature Engineering:** category_encoders
- **Model Interpretability:** SHAP
- **Statistical Analysis:** SciPy

## Key Findings

### Feature Importance (from XGBoost & SHAP Analysis)

Top factors influencing employee turnover:
1. **JoiningYear** - Most significant predictor
2. **PaymentTier** - Second most important factor
3. **Gender** - Strong influence on turnover decisions
4. **City** - Location plays a notable role
5. **Education** - Educational background matters

### Statistical Insights
- All features showed statistical significance in relation to employee turnover
- Chi-square tests for categorical variables and t-tests for numerical variables confirmed feature relevance
- Class imbalance addressed through BorderlineSMOTE improved model performance significantly

### Model Insights
- Tree-based models (XGBoost, Decision Tree) outperformed linear models
- SMOTE resampling improved recall for the minority class (employees leaving)
- XGBoost provided the best balance between precision and recall
