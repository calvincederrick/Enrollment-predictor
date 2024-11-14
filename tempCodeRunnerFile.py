import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the dataset
file_path = r"C:\Users\DERRICK_CALVINCE\Desktop\PR\python projects\EnrollTrack\student-por.csv"  # Replace with actual file path
data = pd.read_csv(file_path)

# Display basic information about the data
print(data.head())
print(data.info())

# Preprocess the Attendance_rate (convert percentage to numerical value)
data['Attendance_rate'] = data['Attendance_rate'].str.rstrip('%').astype('float')  # Remove '%' and convert to float

# Define numerical and categorical features
numerical_features = ['Age', 'GPA', 'Study_Hours', 'Attendance_rate']  # Include Attendance_rate as numerical
categorical_features = ['Gender', 'Major', 'socioeconomic_Status']  # Include Socioeconomic_Status as categorical

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define target and features
X = data.drop(columns=['Student_ID', 'Enrollment_Status'])  # Exclude Student_ID and target column
y = data['Enrollment_Status']  # Use the correct column name

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
joblib.dump(model, 'student_enrollment_model.joblib')
