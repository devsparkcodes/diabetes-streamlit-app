# Diabetes Prediction & Analysis App

A Streamlit-based machine learning application for diabetes prediction and exploratory data analysis. The project uses Logistic Regression and K-Nearest Neighbors (KNN) models and provides an interactive interface for exploring the dataset and generating prediction results.

---

## Project Overview

This project combines exploratory data analysis, data preprocessing, machine learning, and interactive visualization into a single Streamlit application.

The application allows users to explore the diabetes dataset, analyze relationships between different features, and use trained classification models to generate diabetes predictions based on entered health information.

---

## Features

- Dataset overview
- Dataset shape and statistical summary
- Missing-value analysis
- Exploratory Data Analysis (EDA)
- Count plots
- Box plots
- KDE plots
- Correlation heatmap
- Diabetes prediction using Logistic Regression
- Diabetes prediction using K-Nearest Neighbors (KNN)
- User-friendly prediction interface
- Prediction probability and result display

---

## How It Works

The application follows a machine learning workflow:

```text
Diabetes Dataset
      ↓
Data Exploration
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Preparation
      ↓
Machine Learning Models
      ↓
Prediction
      ↓
Prediction Probability & Result
```

The application provides two classification approaches:

- Logistic Regression
- K-Nearest Neighbors (KNN)

---

## Dataset

The project uses `diabetes_prediction_dataset.csv`.

The dataset contains health and demographic information used for analysis and prediction, including:

| Feature | Description |
|---|---|
| `gender` | Gender of the patient |
| `age` | Age of the patient |
| `hypertension` | Hypertension status |
| `heart_disease` | Heart disease status |
| `smoking_history` | Smoking history |
| `bmi` | Body Mass Index |
| `HbA1c_level` | HbA1c level |
| `blood_glucose_level` | Blood glucose level |
| `diabetes` | Diabetes outcome |

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application and machine learning development |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| Scikit-Learn | Machine learning and model development |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |

---

## Project Structure

```text
diabetes-streamlit-app/
│
├── assets/
│
├── app.py
├── diabetes_prediction_dataset.csv
├── requirements.txt
├── .gitignore
└── README.md
```

> The exact repository structure may vary depending on the current project files.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/devsparkcodes/diabetes-streamlit-app.git
```

### Navigate to the Project

```bash
cd diabetes-streamlit-app
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

**Windows:**

```powershell
venv\Scriptsctivate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open through the local Streamlit server.

---

## Learning Outcomes

Through this project, I practiced:

- Data exploration and analysis
- Data preprocessing
- Exploratory Data Analysis
- Feature preparation
- Classification models
- Logistic Regression
- K-Nearest Neighbors
- Model prediction
- Data visualization
- Building interactive Streamlit applications

---

## Future Improvements

- Compare additional machine learning models
- Add detailed model performance comparison
- Improve prediction visualizations
- Add model explainability
- Improve the overall user interface
- Add more comprehensive evaluation metrics

---

## Disclaimer

This project is intended for educational and demonstration purposes only. Its predictions should not be considered a medical diagnosis or a substitute for professional medical advice.

---

## Author

**Muhammad Umar**

Building practical applications at the intersection of software engineering and AI.

- GitHub: https://github.com/devsparkcodes
- LinkedIn: https://linkedin.com/in/devsparkcodes
