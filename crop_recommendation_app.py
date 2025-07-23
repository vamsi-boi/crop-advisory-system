# crop_recommendation_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# === PAGE CONFIG ===
st.set_page_config(page_title="🌾 Crop Advisor for Farmers", layout="centered")
st.title("👨‍🌾 Smart Crop Recommendation System")
st.markdown("Helping small farmers choose the right crop based on soil and climate.")

# === LOAD DATASET ===
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

df = load_data()
X = df.drop("label", axis=1)
y = df["label"]

# === TRAIN MODEL ===
@st.cache_resource
def train_model():
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

# === SIDEBAR INPUTS ===
st.sidebar.header("📥 Enter Soil & Climate Info")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90)
P = st.sidebar.slider("Phosphorous (P)", 5, 145, 42)
K = st.sidebar.slider("Potassium (K)", 5, 205, 43)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 24.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 80.0)
ph = st.sidebar.slider("Soil pH", 3.5, 9.5, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 150.0)

user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# === RECOMMENDATION LOGIC ===
if st.sidebar.button("🌱 Recommend Crop"):
    prediction = model.predict(user_input)[0]
    probs = model.predict_proba(user_input)[0]
    top_indices = np.argsort(probs)[-3:][::-1]
    top_crops = model.classes_[top_indices]

    st.subheader("✅ Recommended Crop")
    st.success(f"**{prediction.upper()}** is the best crop for these conditions.")

    st.markdown("### 🌿 Top 3 Suggestions")
    for i, crop in enumerate(top_crops, 1):
        st.write(f"{i}. {crop.title()} ({probs[top_indices[i-1]]*100:.2f}%)")

    # === RISK & ADVISORY REPORT ===
    st.markdown("### ⚠️ Risk & Fertilizer Advisory")
    tips = []

    if rainfall < 50:
        tips.append("🚨 Low rainfall — use drought-tolerant crops like millets or chickpeas.")
    if temperature > 35:
        tips.append("🔥 High temperature — avoid heat-sensitive crops.")
    if temperature < 15:
        tips.append("❄️ Low temperature — frost can damage young crops.")

    if N < 50:
        tips.append("🌾 Nitrogen is LOW — Add Urea or compost manure.")
    if P < 40:
        tips.append("🌾 Phosphorous is LOW — Add SSP or bone meal.")
    if K < 40:
        tips.append("🌾 Potassium is LOW — Use MOP or wood ash.")

    if ph < 5.5 or ph > 7.5:
        tips.append("⚗️ Soil pH is unbalanced — Add lime (acidic) or gypsum (alkaline).")

    if humidity > 90:
        tips.append("💧 High humidity — Fungal infections like mildew are more likely.")

    if tips:
        for t in tips:
            st.info(t)
    else:
        st.success("Soil and climate conditions are optimal. Good to go! 🚜")

# === ANALYTICS CHARTS ===
st.markdown("---")
st.header("📈 Crop Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("NPK Levels")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df["N"], kde=True, color="skyblue", label="Nitrogen")
    sns.histplot(df["P"], kde=True, color="lightgreen", label="Phosphorous")
    sns.histplot(df["K"], kde=True, color="orange", label="Potassium")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Crop Frequency")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    crop_counts = df["label"].value_counts().head(10)
    sns.barplot(x=crop_counts.values, y=crop_counts.index, palette="viridis")
    plt.xlabel("Count")
    plt.title("Top Crops in Dataset")
    st.pyplot(fig2)

# === FOOTER ===
st.markdown("---")
st.caption("🧠 Developed using Python, Streamlit, and Machine Learning | Dataset: Kaggle")
