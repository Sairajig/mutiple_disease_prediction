# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# Step 1: Load the dataset
# Ensure the dataset is in the "dataset" folder with the correct name
dataset_path = "dataset/heart.csv"  # Update this if your file is in another location

if not os.path.exists(dataset_path):
    print(f"Error: Dataset file not found at {dataset_path}. Please check the file path.")
    exit()

data = pd.read_csv(dataset_path)
print("Dataset loaded successfully!\n")

# Step 2: Explore the dataset
print("First 5 rows of the dataset:")
print(data.head(), "\n")

print("Dataset Information:")
print(data.info(), "\n")

print("Summary Statistics:")
print(data.describe(), "\n")

# Step 3: Preprocess the data
# Separate features (X) and target (y)
X = data.iloc[:, :-1]  # Features: all columns except the last one
y = data.iloc[:, -1]   # Target: the last column

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Feature scaling completed.\n")

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training Set: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing Set: X_test: {X_test.shape}, y_test: {y_test.shape}\n")

# Step 5: Train the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, max_depth=None, min_samples_split=2
)
rf_model.fit(X_train, y_train)
print("Model training completed.\n")

# Step 6: Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix, "\n")
print("Classification Report:")
print(class_report, "\n")

# Step 7: Save the trained model
output_dir = "saved_models"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
model_path = os.path.join(output_dir, "heart_disease.pkl")

with open(model_path, "wb") as file:
    pickle.dump(rf_model, file)
print(f"Model saved successfully at {model_path}")
