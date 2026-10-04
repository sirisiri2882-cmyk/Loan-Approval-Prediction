# 🏦 Loan Approval Prediction Using Deep Learning

A machine learning web application that uses an **Artificial Neural Network (ANN)** built with TensorFlow to estimate loan approval probability based on applicant information.

## 🎯 Project Overview

This project demonstrates how deep learning can be applied to a loan prediction dataset. Users enter applicant details through an interactive Streamlit interface, and the trained neural network produces an estimated approval probability.

## ✨ Features

* Data cleaning and missing-value handling
* Categorical feature encoding
* Feature scaling
* Artificial Neural Network built with TensorFlow
* Model evaluation using accuracy, classification report, and confusion matrix
* Interactive Streamlit web application
* Visual prediction results and probability display

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Pandas and NumPy
* Scikit-learn
* Streamlit
* Matplotlib and Seaborn

## 🧠 Model Workflow

1. Load the loan application dataset.
2. Handle missing values.
3. Encode categorical variables and scale features.
4. Split the dataset into training and testing sets.
5. Train the neural network.
6. Evaluate performance on held-out test data.
7. Use the saved model to generate predictions through Streamlit.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Loan-Approval-Prediction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

## 📊 Model Evaluation

The model was evaluated on a held-out test set. The initial test accuracy was approximately 83%; verify the final evaluation results before treating this figure as definitive.

## ⚠️ Disclaimer

This project is for educational purposes only. Predictions are estimates from a trained model and must not be used as actual lending or financial decisions.

## 👩‍💻 Author

Developed as a deep learning project to explore neural networks and interactive machine learning applications.
