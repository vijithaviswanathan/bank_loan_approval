# 🏦 Loan Approval Prediction

A complete **Machine Learning project for predicting loan approval outcomes** based on an applicant’s financial, demographic, employment, credit-history, and property-related information.

The project follows the complete ML workflow from **data cleaning and exploration to feature engineering, class balancing, model training, evaluation, comparison, and model saving**.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Joblib / Pickle

---

## 📥 Dataset

The datasets used for this project are available on Kaggle.

Loan Prediction Dataset 
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Training Dataset 
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?select=train_u6lujuX_CVtuZ9i.csv

Download the required CSV files from Kaggle and place them inside:

```text
data/raw/
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <your-github-repository-url>
cd Loan_Prediction
```

### 2️⃣ Create the Required Folders

```bash
mkdir -p data/raw artifacts figures models
```

### 3️⃣ Download the Dataset

Download the required datasets from Kaggle and place the CSV files inside:

```text
data/raw/
```

### 4️⃣ Install Required Packages

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Main Script

Run the complete machine-learning pipeline using:

```bash
python Main.py
```

---

## 📊 Project Workflow

```text
Raw Data
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Class Balancing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Model Selection
   ↓
Model Saving
```

---

## 🏆 Key Results

- Best Model: Random Forest
- Validation Accuracy: 81.04%
- ROC-AUC: 0.904
- Models Evaluated: 13 ML/DL models
- Top-3 Ensemble Accuracy: 78.67%

---
