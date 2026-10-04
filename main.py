
import pandas as pd

# Load the loan training dataset
data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Display column names
print("\nDataset columns:")
print(data.columns.tolist())

# Display dataset size
print("\nDataset shape:")
print(data.shape)

# Check missing values in each column
print("\nMissing values in each column:")
print(data.isnull().sum())

# Check the loan approval counts
print("\nLoan approval distribution:")
print(data["Loan_Status"].value_counts())

# Check missing values in all columns
print("\nMissing values in all columns:")
print(data.isnull().sum())

# Display total missing values
print("\nTotal missing values:")
print(data.isnull().sum().sum())

# Show only columns that contain missing values
print("\nColumns with missing values:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# Fill missing values in categorical columns with the most common value
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Fill missing values in numerical columns with the median
numerical_columns = [
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nTotal missing values after cleaning:")
print(data.isnull().sum().sum())

# Fill missing values in categorical columns with the most common value
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Fill missing values in numerical columns with the median
numerical_columns = [
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nTotal missing values after cleaning:")
print(data.isnull().sum().sum())

# Fill missing values in categorical columns with the most common value
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Fill missing values in numerical columns with the median
numerical_columns = [
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nTotal missing values after cleaning:")
print(data.isnull().sum().sum())

# Separate input features and target
X = data.drop(["Loan_ID", "Loan_Status"], axis=1)
y = data["Loan_Status"]

# Display the input features and target
print("\nInput features shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nInput feature columns:")
print(X.columns.tolist())

# Convert categorical text columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Display the processed data
print("\nFeatures after converting text to numbers:")
print(X.head())

print("\nShape after conversion:")
print(X.shape)

print("\nChecking for remaining text columns:")
print(X.select_dtypes(include=["object"]).columns.tolist())

from sklearn.model_selection import train_test_split

# Convert loan approval labels into numbers
y = y.map({"N": 0, "Y": 1})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the sizes of the datasets
print("\nTraining features shape:")
print(X_train.shape)

print("\nTesting features shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)

from sklearn.preprocessing import StandardScaler

# Create the feature scaler
scaler = StandardScaler()

# Learn scaling values from training data only
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to testing data
X_test_scaled = scaler.transform(X_test)

# Display the results
print("\nScaled training data shape:")
print(X_train_scaled.shape)

print("\nScaled testing data shape:")
print(X_test_scaled.shape)

print("\nFirst 5 rows of scaled training data:")
print(X_train_scaled[:5])

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Build the neural network
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Configure the model for training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display the neural network architecture
print("\nNeural Network Architecture:")
model.summary()

# Train the neural network
print("\nTraining the neural network...")

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\nModel training completed!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate the model on unseen test data
print("\nEvaluating the model...")

test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predict loan approval probabilities
y_probability = model.predict(X_test_scaled, verbose=0)

# Convert probabilities into 0 or 1 predictions
y_pred = (y_probability >= 0.5).astype(int).ravel()

# Display detailed evaluation results
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained neural network
model.save("loan_approval_model.keras")

print("\nModel saved successfully!")

# Save the feature scaler and the model's input column order
import joblib

joblib.dump(scaler, "loan_scaler.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("Scaler and feature columns saved successfully!")