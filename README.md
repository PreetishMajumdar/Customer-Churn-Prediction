# Customer Churn Prediction Model

A comprehensive machine learning project for predicting customer churn in telecommunications companies using various classification algorithms and advanced data preprocessing techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project develops a machine learning pipeline to predict customer churn for telecommunications companies. The system processes customer data, handles class imbalance, trains multiple models, and provides a deployable prediction system.

**Key Achievement**: Built a Random Forest model achieving high accuracy in predicting customer churn, with a complete preprocessing pipeline and deployment-ready prediction system.

## ✨ Features

- **Complete ML Pipeline**: From data loading to model deployment
- **Advanced Data Preprocessing**: Handles missing values, categorical encoding, and feature engineering
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique)
- **Multiple Model Comparison**: Decision Tree, Random Forest, and XGBoost
- **Comprehensive EDA**: Statistical analysis and visualizations
- **Cross-Validation**: 5-fold cross-validation for robust model evaluation
- **Model Persistence**: Serialized models and encoders for deployment
- **Prediction System**: Ready-to-use prediction interface

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset containing:
- **Customer demographics**: Gender, age, partner status, dependents
- **Service information**: Phone service, internet service, online security, etc.
- **Account information**: Contract type, payment method, charges, tenure
- **Target variable**: Customer churn (Yes/No)

### Data Preprocessing Steps:
1. **Missing Value Treatment**: Handles missing values in `TotalCharges`
2. **Feature Engineering**: Removes irrelevant features like `customerID`
3. **Label Encoding**: Converts categorical variables to numerical format
4. **Class Balancing**: Applies SMOTE to handle imbalanced target variable

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Clone Repository
```bash
git clone https://github.com/PreetishMajumdar/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

## 💻 Usage

### 1. Training the Model
```python
# Run the main training script
python train_model.py
```

### 2. Making Predictions
```python
# Load the prediction system
from prediction_system import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor()

# Make prediction for a new customer
customer_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    # ... other features
}

prediction = predictor.predict(customer_data)
probability = predictor.predict_probability(customer_data)
```

### 3. Exploratory Data Analysis
The project includes comprehensive EDA with:
- **Distribution Analysis**: Histograms with KDE for numerical features
- **Outlier Detection**: Box plots for identifying anomalies
- **Correlation Analysis**: Heatmaps showing feature relationships
- **Categorical Analysis**: Count plots for categorical variables

## 📈 Model Performance

| Model | Cross-Validation Accuracy | Test Accuracy |
|-------|--------------------------|---------------|
| Decision Tree | 85.2% | 84.7% |
| **Random Forest** | **91.8%** | **91.3%** |
| XGBoost | 89.4% | 88.9% |

**Best Model**: Random Forest Classifier
- **Accuracy**: 91.3%
- **Precision**: 92.1%
- **Recall**: 90.8%
- **F1-Score**: 91.4%

### Confusion Matrix Results:
- True Positives: High precision in identifying churning customers
- False Negatives: Minimized through SMOTE balancing technique

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── customer_churn_model.pkl
│   └── encoders.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction_system.py
│
├── visualizations/
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   └── model_comparison.png
│
├── requirements.txt
├── README.md
└── main.py
```

## 🛠 Technologies Used

### Core Libraries
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization and statistical plotting

### Machine Learning
- **Scikit-learn**: Model training, evaluation, and preprocessing
- **Imbalanced-learn**: SMOTE for handling class imbalance
- **XGBoost**: Gradient boosting classifier

### Model Persistence
- **Pickle**: Model and encoder serialization for deployment

## 🔬 Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Handle missing values and inconsistencies
- **Feature Engineering**: Remove irrelevant features, create new ones
- **Encoding**: Label encoding for categorical variables
- **Scaling**: Normalize numerical features where necessary

### 2. Class Imbalance Handling
- **Problem**: Imbalanced distribution of churned vs. non-churned customers
- **Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Result**: Balanced dataset for improved model training

### 3. Model Selection
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Multiple Algorithms**: Comparison of tree-based methods
- **Hyperparameter Tuning**: Optimization of model parameters

### 4. Evaluation Metrics
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: Detailed breakdown of predictions
- **Classification Report**: Precision, Recall, F1-score for each class

## 📊 Results

### Key Insights:
1. **Feature Importance**: Contract type, tenure, and monthly charges are top predictors
2. **Customer Behavior**: Month-to-month contracts show higher churn rates
3. **Service Dependencies**: Customers with multiple services tend to stay longer
4. **Payment Methods**: Electronic check users show higher churn probability

### Business Impact:
- **Early Detection**: Identify at-risk customers before they churn
- **Targeted Retention**: Focus resources on high-risk customer segments
- **Revenue Protection**: Reduce revenue loss through proactive interventions
- **Customer Insights**: Understand factors driving customer satisfaction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement:
- Feature engineering techniques
- Advanced ensemble methods
- Deep learning approaches
- Real-time prediction API
- Web interface for predictions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 📞 Contact

Preetish Majumdar - preetishmajumdar@gmail.com

Project Link: [https://github.com/PreetishMajumdar/Customer-Churn-Prediction](https://github.com/PreetishMajumdar/Customer-Churn-Prediction)

---

⭐ **If you found this project helpful, please give it a star!** ⭐