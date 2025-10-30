import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Load model and scaler ---
model = load_model("tata_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# --- Streamlit page setup ---
st.set_page_config(page_title="TATA Motors Stock Prediction", layout="wide")
st.title("📈 TATA Motors Stock Price Prediction")

# --- Load latest data ---
data = yf.download('TATAMOTORS.NS', start='2015-01-01')
close_data = data[['Close']].values

# --- Normalize ---
scaled_data = scaler.transform(close_data)

# --- Prepare input for last 60 days ---
time_step = 60
X_input = scaled_data[-time_step:]
X_input = np.reshape(X_input, (1, time_step, 1))

# --- Predict next day price ---
predicted_price = model.predict(X_input)
predicted_price = scaler.inverse_transform(predicted_price)

# --- Display predicted price ---
st.markdown(f"### 🪙 Predicted Next Day Close Price: ₹{predicted_price[0][0]:.2f}")

# --- Reconstruct prediction for visualization ---
X_test = []
y_test = []
for i in range(time_step, len(scaled_data)):
    X_test.append(scaled_data[i-time_step:i, 0])
    y_test.append(scaled_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# --- Predict full test data ---
predicted_full = model.predict(X_test)
predicted_full = scaler.inverse_transform(predicted_full)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- Plot actual vs predicted ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test_actual, label='Actual Price', color='blue')
ax.plot(predicted_full, label='Predicted Price', color='red')
ax.set_title("TATA Motors Stock Price Prediction (Historical)")
ax.legend()
st.pyplot(fig)
