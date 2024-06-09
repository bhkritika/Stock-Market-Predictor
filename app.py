import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the trained Gradient Boosting model
model = joblib.load('Stock_Market.pkl')

# Set page configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Set background image and font color
st.markdown(
    """
    <style>
    body {
        background-color: lightblue;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Stock Market Predictor')

# Input for stock symbol and date range
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start_date = pd.Timestamp(st.date_input('Start Date', value=pd.Timestamp('2012-01-01'), min_value=pd.Timestamp('2012-01-01'), max_value=pd.Timestamp('2024-06-09')))
end_date = pd.Timestamp(st.date_input('End Date', value=pd.Timestamp('2024-06-09'), min_value=pd.Timestamp('2012-01-01'), max_value=pd.Timestamp('2024-06-09')))

# Fetching the data
data = yf.download(stock, start=start_date, end=end_date)
data.reset_index(inplace=True)

st.subheader('Stock Data')

# Filter data based on selected date range
selected_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

if selected_data.empty:
    st.write("No data available for the selected date range.")
else:
    st.write(selected_data)

# Splitting the data into training and testing sets
data_train = data.iloc[:int(len(data) * 0.80)]
data_test = data.iloc[int(len(data) * 0.80):]

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train[['Open', 'High', 'Low', 'Volume', 'Close']])
data_test_scaled = scaler.transform(data_test[['Open', 'High', 'Low', 'Volume', 'Close']])

# Plotting Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(ma_50_days, 'r')
ax1.plot(data.Close, 'g')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.set_title('Price vs MA50')
ax1.legend(['MA50', 'Price'], loc='best')
st.pyplot(fig1)

# Plotting Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(ma_50_days, 'r')
ax2.plot(ma_100_days, 'b')
ax2.plot(data.Close, 'g')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.set_title('Price vs MA50 vs MA100')
ax2.legend(['MA50', 'MA100', 'Price'], loc='best')
st.pyplot(fig2)

# Plotting Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(ma_100_days, 'r')
ax3.plot(ma_200_days, 'b')
ax3.plot(data.Close, 'g')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.set_title('Price vs MA100 vs MA200')
ax3.legend(['MA100', 'MA200', 'Price'], loc='best')
st.pyplot(fig3)

# Preparing the data for prediction
x_test = data_test_scaled[:, :-1]  # All columns except 'Close'
y_test = data_test_scaled[:, -1]   # Only 'Close' column

# Predicting with the Gradient Boosting model
predictions = model.predict(x_test)

# Inversing the scaling
y_test_actual = y_test * (1 / scaler.scale_[-1])
predictions_actual = predictions * (1 / scaler.scale_[-1])

# Plotting the Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.plot(y_test_actual, 'g', label='Original Price')
ax4.plot(predictions_actual, 'r', label='Predicted Price', linestyle='--')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
ax4.set_title('Comparison of Predicted and Original Prices')
st.pyplot(fig4)
