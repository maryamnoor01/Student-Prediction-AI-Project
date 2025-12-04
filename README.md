# Student Gender Prediction using Support Vector Classifier (SVC)

## 1. Overview

This project implements a machine learning solution to predict a
student's **Gender** based on a comprehensive dataset of student
performance and demographic factors. The prediction model is built using
a **Support Vector Classifier (SVC)** and is deployed using a simple
**Flask** web application for real-time inference.

The goal of this project is to demonstrate the end-to-end process of
data preprocessing, model training, evaluation, saving, and deployment.

## 2. Project Structure

The repository contains the following key files:

-   **`project.ipynb`**: A Jupyter Notebook detailing the entire machine
    learning workflow, including:
    -   Data Loading and Initial Exploration.
    -   Data Preprocessing (Handling missing values, data type
        conversion, feature dropping).
    -   Train-Test Split.
    -   Model Training using `sklearn.svm.SVC`.
    -   Model Evaluation (Accuracy Score).
    -   Model persistence using `pickle`.
-   **`app.py`**: The Flask application for serving the trained machine
    learning model. It handles user input, performs predictions, and
    displays results.
-   **`model_svc.pkl`**: The trained Support Vector Classifier model,
    serialized using `pickle` for deployment (created by
    `project.ipynb`).
-   **`Student_performance_data_.csv`**: The dataset used for training
    the model.
-   **`templates/`**: Directory containing HTML templates for the Flask
    application (e.g., `index.html`, `results.html`, `error.html`).

## 3. Technology Stack

-   **Language:** Python
-   **Core Libraries:** `pandas`, `numpy`, `scikit-learn`
-   **Model:** Support Vector Classifier (SVC)
-   **Deployment:** Flask
-   **Environment Management:** `pickle` for model serialization

## 4. Dataset

The dataset used is sourced from Kaggle:\
https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

**Key Features include:**

-   `Age`
-   `Ethnicity`
-   `ParentalEducation`
-   `StudyTimeWeekly`
-   `Absences`
-   `Tutoring`
-   `ParentalSupport`
-   `Sports`
-   `Music`
-   `Volunteering`
-   `GPA`
-   `GradeClass`
-   **Target Variable:** `Gender`

## 5. Machine Learning Workflow

### Data Preparation

1.  Exploration: Checked dataset shape (2392 rows, 15 columns) and
    verified that there are no missing values.
2.  Cleaning: The column `Extracurricular` was dropped.
3.  Type Conversion: The columns `StudyTimeWeekly`, `GPA`, and
    `GradeClass` were converted to integer.
4.  Feature Engineering: Independent features (`X`) were separated from
    the target variable (`y`).

### Model Training and Evaluation

1.  Data Split: 70% training and 30% testing.
2.  Model: Support Vector Classifier (SVC).
3.  Prediction: Gender predicted on test data.
4.  Performance: Accuracy achieved was **49.4%**.
5.  Model Saving: Trained model saved as `model_svc.pkl`.

## 6. Deployment (Flask)

The Flask application performs the following: 1. Loads the trained
model. 2. Renders the input form. 3. Accepts student data. 4. Predicts
Gender. 5. Displays results.

## 7. Setup and Installation

### Step 1: Clone the Repository

``` bash
git clone <repository_url>
cd <project_directory>
```

### Step 2: Install Dependencies

``` bash
pip install pandas numpy scikit-learn flask
```

### Step 3: Run the Notebook

Run all cells in `project.ipynb` to train the model.

### Step 4: Run Flask Application

``` bash
python app.py
```

The application will run at:

    http://127.0.0.1:5000/

------------------------------------------------------------------------

© Student Gender Prediction using Machine Learning
