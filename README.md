# 🩺 Diabetes Prediction & Analysis App

This is a **Streamlit web application** that allows users to predict the probability of diabetes based on personal health information. The app uses **Logistic Regression** and **K-Nearest Neighbors (KNN)** models and also provides **exploratory data analysis (EDA)** and data visualization features.

---

## Features

- Dataset overview with shape, missing values, and statistical summary
- Exploratory Data Analysis (EDA) with:
  - Count plots
  - Box plots
  - KDE plots
  - Correlation heatmap
- Predict diabetes using:
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
- Enter patient information in a user-friendly interface
- View prediction probability and results

---

## Dataset

The dataset `diabetes_prediction_dataset.csv` contains the following columns:

- `gender`: Gender of the patient
- `age`: Age of the patient
- `hypertension`: Hypertension status (Yes/No)
- `heart_disease`: Heart disease status (Yes/No)
- `smoking_history`: Smoking history
- `bmi`: Body Mass Index
- `HbA1c_level`: HbA1c Level
- `blood_glucose_level`: Blood Glucose Level
- `diabetes`: Diabetes outcome (0 = No, 1 = Yes)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/diabetes-streamlit-app.git
cd diabetes-streamlit-app
```

2. Create a virtual environment (optional but recommended):

```python -m venv venv```

4. Activate the virtual environment:

Windows:
```venv\Scripts\activate```

Mac/Linux:
```source venv/bin/activate```

Install dependencies:
```pip install -r requirements.txt```

Run the app:
```streamlit run app.py```

Open your browser and go to the local URL shown in the terminal.

### Requirements
Create a requirements.txt file with the following packages:

numpy
pandas
streamlit
seaborn
matplotlib
scikit-learn

Optional: For exact versions (recommended for reproducibility):

pip freeze > requirements.txt

### .gitignore
Create a .gitignore file to avoid unnecessary files:

__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
.ipynb_checkpoints/

### Author

Muhammad Umar
Data Science Enthusiast & Python Developer

### License

This project is licensed under the MIT License.
