# Enrollment-predictor

# EnrollTrack: Student Enrollment Status Prediction

## Overview

**EnrollTrack** is a machine learning project that predicts the enrollment status of students based on various features such as age, gender, GPA, study hours, attendance rate, and socioeconomic status. The model is trained on a dataset of student information and uses classification techniques to predict whether a student is "Enrolled" or in "Need Support".

## Features

- **Student_ID**: Unique identifier for each student
- **Age**: Age of the student
- **Gender**: Gender of the student
- **Major**: The student's major
- **GPA**: Grade Point Average of the student
- **Study_Hours**: Number of hours the student spends studying per week
- **Attendance_rate**: The attendance rate of the student
- **Socioeconomic_Status**: The socioeconomic status of the student (e.g., Low, Medium, High)
- **Enrollment_Status**: Target variable (Enrolled or Needs Support)

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.7 or higher
- Pip (Python's package installer)

### Installation

1. Clone the repository or download the project files.
2. Navigate to the project directory in the terminal.

3. Install the required libraries:
   ```bash
   pip install pandas scikit-learn

 Running the Code
Place your dataset (in CSV format) in the project directory.

Run the Python script to train the model and make predictions
The script will output the following:

Accuracy score
Classification report including precision, recall, and F1-score

Accuracy: 1.0
Classification Report:
                precision    recall  f1-score   support
    Enrolled       1.00      1.00      1.00         6
Needs Support       1.00      1.00      1.00         2
    accuracy                           1.00         8
   macro avg       1.00      1.00      1.00         8
 weighted avg       1.00      1.00      1.00         8

