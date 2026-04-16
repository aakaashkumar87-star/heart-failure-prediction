import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# TITLE
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title(" Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/heart.csv")

# -----------------------------
# ENCODE DATA
# -----------------------------
df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# -----------------------------
# SPLIT DATA
# -----------------------------
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Enter Patient Details")

def user_input():
    age = st.sidebar.slider("Age", 20, 80, 40)
    sex = st.sidebar.selectbox("Sex", ["M", "F"])
    chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.sidebar.slider("Resting BP", 80, 200, 120)
    cholesterol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

    input_dict = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    return pd.DataFrame([input_dict])

input_df = user_input()

# -----------------------------
# ENCODE INPUT DATA
# -----------------------------
input_encoded = input_df.copy()

for col in input_encoded.select_dtypes(include='object').columns:
    input_encoded[col] = LabelEncoder().fit_transform(input_encoded[col])

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(input_encoded)

st.subheader("Prediction Result")

if prediction[0] == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk of Heart Disease")

# -----------------------------
# MODEL ACCURACY
# -----------------------------
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# SHOW DATASET
# -----------------------------
if st.checkbox("Show Dataset"):
    st.write(df.head())