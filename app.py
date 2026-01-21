# ============================== IMPORTS ==============================
import numpy as np
import pandas as pd

import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================== PAGE CONFIG ==============================
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("🩺 Diabetes Prediction & Analysis App")

# ============================== LOAD DATA ==============================
df = pd.read_csv("diabetes_prediction_dataset.csv")

# ==========>> Cleaning Process <<==========
cat_cols = ["gender", "smoking_history"]
for col in cat_cols:
    df[col] = df[col].str.strip().str.lower()
    df[col] = df[col].fillna(df[col].mode()[0])

nums_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
for col in nums_cols:
    df[col] = pd.to_numeric(df[col])
    df[col] = df[col].fillna(df[col].median())

# Copy DataSet
df_vis = df.copy()

# Convert Diabetes Column into Categorical Column for visulization
df_vis["diabetes"] = df_vis["diabetes"].map({
    0: "no",
    1: "yes"
})

# Convert hypertension and heart_disaase column into Categorcal Column
binary_cols = ["hypertension", "heart_disease"]
for col in binary_cols:
    df[col] = df[col].map({
        0: "No",
        1: "Yes"
    })
cat_cols = cat_cols + binary_cols

# ============================== TABS ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "📈 EDA",
    "🤖 Logistic Regression",
    "📍 KNN"
])

# ============================== TAB 1 ==============================
with tab1:
    st.subheader("📁 Dataset Overview")
    st.write("#### Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Shape")
        st.write(df.shape)
        st.write("**100000 Rows**")
        st.write("**9 Columns**")

    with col2:
        st.write("#### Missing Values")
        st.dataframe(df.isnull().sum())

    st.write("#### Statistical Summary")
    st.dataframe(df.describe())

# ============================== TAB 2 (EDA) ==============================
with tab2:
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    # Count Plot (Diabetes Distribution by Gender)
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df_vis, x="gender", hue="diabetes", ax=ax)
        ax.set_title("Diabetes Distribution by Gender")
        st.pyplot(fig)

    # Count Plot (Distribution of Diabetes Classes)
    with col2:
        fig, ax = plt.subplots()
        sns.countplot(data=df_vis, x="diabetes", ax=ax)
        ax.set_title("Distribution of Diabetes Classes")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    
    # Box Plot (BMI Comparison Between Diabetic and Non-Diabetic Patients)
    with col3:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_vis, x="diabetes", y="bmi", ax=ax)
        ax.set_title("BMI Comparison Between Diabetic and Non-Diabetic Patients")
        st.pyplot(fig)

    # Box Plot (HbA1c Level Distribution by Diabetes Outcome)
    with col4:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_vis, x="diabetes", y="HbA1c_level", ax=ax)
        ax.set_title("HbA1c Level Distribution by Diabetes Outcome")
        st.pyplot(fig)

    col5, col6 = st.columns(2)

    # Box Plot (Blood Glucose Level by Diabetes Status)
    with col5:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_vis, x="diabetes", y="blood_glucose_level", ax=ax)
        ax.set_title("Blood Glucose Level by Diabetes Status")
        st.pyplot(fig)

    # KDE Plot (Distribution of Patient Age)
    with col6:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df_vis, x="age", hue="diabetes", fill=True, ax=ax)
        ax.set_title("Distribution of Patient Age")
        st.pyplot(fig)

    col7, col8 = st.columns(2)

    # Count Plot (Impact of Smoking History on Diabetes)
    with col7:
        fig, ax = plt.subplots()
        sns.countplot(data=df_vis, x="smoking_history", hue="diabetes", ax=ax)
        ax.set_title("Impact of Smoking History on Diabetes")
        st.pyplot(fig)
    
    # Heatmap (Correlation Heatmap of Numerical Features)
    with col8:
        fig, ax = plt.subplots()
        sns.heatmap(df_vis.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap of Numerical Features")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=18, fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=60, fontsize=9)
        st.pyplot(fig)


# ==============================>> MODEL TRAINING <<==============================
# =============>> Training Process <<=============
# ====>> Define Features and Target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# =============== Model Pipeline ===============
# ====>> Numerical Pipeline
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ====>> Categorical Pipeline
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

# ====>> ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, nums_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ====>> Data Spliting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====>> Final Pipelines (Preprocessing + Model)
log_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

knn_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=15))
])

# ====>> Model Training
log_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# ============================== TAB 3 ==============================
with tab3:
    st.subheader("📊 Logistic Regression Model Performance")

    log_y_pred = log_model.predict(X_test)

    # Accuracy
    log_acc = accuracy_score(y_test, log_y_pred)
    # Show accuracy as metric
    st.metric(
        label="Model Accuracy",
        value=f"{log_acc*100:.2f}%"
    )

    # Classification report
    log_report = classification_report(y_test, log_y_pred, output_dict=True)
    log_report_df = pd.DataFrame(log_report).transpose()
    # detailed report
    with st.expander("📋 Show Classification Report"):
        st.dataframe(log_report_df.style.format("{:.2f}"))

    # ============================== USER INPUT FUNCTION ==============================
    st.subheader("🧑 Enter Patient Information")

    gender = st.selectbox("Gender", df["gender"].unique(), key="log_gender")
    age = st.number_input("Age", 1, 120, 30, key="log_age")
    hypertension = st.selectbox("Hypertension", ["Yes", "No"], key="log_hypertension")
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key="log_heart")
    smoking = st.selectbox("Smoking History", df["smoking_history"].unique(), key="log_smoking")
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, key="log_bmi")
    hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, key="log_hba1c")
    glucose = st.number_input("Blood Glucose Level", 50.0, 300.0, 120.0, key="log_glucose")

    if st.button("🔮 Predict Diabetes (Logistic Regression)"):
        log_user_df = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [glucose]
        })

        # =====================>> PREDICTION <<=====================
        log_user_prd = log_model.predict(log_user_df)

        log_prob = log_model.predict_proba(log_user_df)[0][1]

        st.subheader("📌 Prediction Result")
        st.write(f"Diabetes Probability: **{log_prob*100:.2f}%**")

        if log_user_prd[0] == 1:
            st.error("❌ Diabetic")
        else:
            st.success("✅ Not Diabetic")

# ============================== TAB 4 ==============================
with tab4:
    st.subheader("📊 KNN Model Performance")

    knn_y_pred = knn_model.predict(X_test)

    # Accuracy
    knn_acc = accuracy_score(y_test, knn_y_pred)
    # Show accuracy as metric
    st.metric(
        label="Model Accuracy",
        value=f"{knn_acc*100:.2f}%"
    )

    # Classification report
    knn_report = classification_report(y_test, knn_y_pred, output_dict=True)
    knn_report_df = pd.DataFrame(knn_report).transpose()
    # detailed report
    with st.expander("📋 Show Classification Report"):
        st.dataframe(knn_report_df.style.format("{:.2f}"))

    # ============================== USER INPUT FUNCTION ==============================
    st.subheader("🧑 Enter Patient Information")

    gender = st.selectbox("Gender", df["gender"].unique(), key="knn_gender")
    age = st.number_input("Age", 1, 120, 30, key="knn_age")
    hypertension = st.selectbox("Hypertension", ["Yes", "No"], key="knn_hypertension")
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key="knn_heart")
    smoking = st.selectbox("Smoking History", df["smoking_history"].unique(), key="knn_smoking")
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, key="knn_bmi")
    hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, key="knn_hba1c")
    glucose = st.number_input("Blood Glucose Level", 50.0, 300.0, 120.0, key="knn_glucose")

    if st.button("🔮 Predict Diabetes (KNN)"):
        knn_user_df = pd.DataFrame({
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [glucose]
        })

        # =====================>> PREDICTION <<=====================
        knn_user_prd = knn_model.predict(knn_user_df)

        knn_prob = knn_model.predict_proba(knn_user_df)[0][1]

        st.subheader("📌 Prediction Result")
        st.write(f"Diabetes Probability: **{knn_prob*100:.2f}%**")

        if knn_user_prd[0] == 1:
            st.error("❌ Diabetic")
        else:
            st.success("✅ Not Diabetic")