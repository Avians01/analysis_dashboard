# pv_dashboard.py
# streamlit run pv_dashboard.py
# cd "D:\truxco energy dataset\analysis of pv data"

import os
import streamlit as st
import pandas as pd
import numpy as np 
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="PV Dashboard")
st.title("PV Data Analysis Dashboard – PV-001")

import gdown
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    file_id = "1fNYbatRAITaYO0lkicQsyK00Y2vDWD59"
    url = f"https://drive.google.com/uc?id={file_id}"
    file_path = "pv-data.csv"
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)
    df = pd.read_csv(file_path, low_memory=False)
    df['datetime'] = pd.to_datetime(df['datetime_millis'], errors='coerce')
    df = df.dropna(subset=['datetime', 'pac'])
    df = df.sort_values('datetime')
    df = df[df['device_id'] == 'PV-001']
    return df

df = load_data()

st.subheader("Data Preview")
st.dataframe(df.head(100))

st.subheader("Data Preview")
st.dataframe(df.head(100))

with st.expander("Basic Statistics"):
    st.write(df[['pac', 'eac_today', 'eac_total']].describe())

st.subheader("Power Output Over Time")
df_hourly = df.set_index('datetime')['pac'].resample('h').mean().interpolate().reset_index()
fig1 = px.line(df_hourly, x='datetime', y='pac', title='Hourly Mean PAC Output – PV001')
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Inverter Status Over Time")
df_status = df.set_index('datetime')[['inverter_status']].resample('h').mean().reset_index()
fig2 = px.line(df_status, x='datetime', y='inverter_status', title='Hourly Mean Inverter Status')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Seasonal Decomposition")
result = seasonal_decompose(df_hourly.set_index('datetime')['pac'], model='additive', period=24)
fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
result.observed.plot(ax=axes[0], title='Observed')
result.trend.plot(ax=axes[1], title='Trend')
result.seasonal.plot(ax=axes[2], title='Seasonal')
result.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
st.pyplot(fig)

st.subheader("Exploratory Data Analysis (EDA)")
tab1, tab2, tab3 = st.tabs(["Temperature Distribution", "Power vs Temperature", "Diurnal Boxplot"])

with tab1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['temp1'], kde=True, ax=ax)
    ax.set_title("Temperature Distribution – temp1")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x='temp1', y='real_output_power', data=df, ax=ax)
    ax.set_title("Power vs Temperature")
    st.pyplot(fig)

with tab3:
    df['hour'] = df['datetime'].dt.hour
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x='hour', y='real_output_power', data=df, ax=ax)
    ax.set_title("Hourly Output Power")
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df[['pac', 'inverter_status', 'temp1', 'temp2', 'temp3']].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader("ADF Stationarity Test")

adf_result = adfuller(df_hourly['pac'].dropna())
st.markdown(f"""
- **ADF Statistic**: `{adf_result[0]:.4f}`  
- **p-value**: `{adf_result[1]:.4f}`  
- **Critical Values**:
""")
for key, value in adf_result[4].items():
    st.markdown(f"   - {key}: `{value:.4f}`")

if adf_result[1] < 0.05:
    st.success("The series is **stationary** (p < 0.05).")
else:
    st.warning("The series is **non-stationary** (p ≥ 0.05).")

st.subheader("SARIMA Forecasting")

df_device = df.copy()
df_device['datetime'] = pd.to_datetime(df_device['datetime'], errors='coerce')
df_device = df_device.set_index('datetime')
df_device_numeric = df_device.select_dtypes(include='number')
df_hourly = df_device_numeric.resample('h').mean().interpolate()
df_hourly['pac_log'] = np.log1p(df_hourly['pac'])

train_log = df_hourly['pac_log'][:'2025-05-29']
test_log = df_hourly['pac_log']['2025-05-30':]
actual = df_hourly['pac']['2025-05-30':]
exog_vars = ['temp2']
exog_train = df_hourly.loc[train_log.index, exog_vars]
exog_test = df_hourly.loc[test_log.index, exog_vars]

model = SARIMAX(train_log, exog=exog_train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

if len(test_log) > 0:
    forecast_log = results.predict(start=test_log.index[0], end=test_log.index[-1], exog=exog_test)
    forecast = np.expm1(forecast_log)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.expm1(train_log[-7*24:]), label='Train')
    ax.plot(actual, label='Actual')
    ax.plot(forecast, label='Forecast')
    ax.set_title("SARIMA Forecast vs Actual – PAC")
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45) 
    fig.autofmt_xdate()
    st.pyplot(fig)

    residuals = actual - forecast
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(residuals)
    ax.set_title("Forecast Residuals")
    ax.tick_params(axis='x', labelrotation=45) 
    fig.autofmt_xdate()   
    st.pyplot(fig)

    # ACF/PACF Lags
    st.subheader("Residual ACF/PACF")
    lag_input = st.slider("Select number of lags (<= 50% of sample size)", min_value=1, max_value=min(40, len(residuals)//2), value=min(20, len(residuals)//2))

    fig1 = plot_acf(residuals.dropna(), lags=lag_input)
    st.pyplot(fig1.figure)

    fig2 = plot_pacf(residuals.dropna(), lags=lag_input)
    st.pyplot(fig2.figure)

    def safe_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def safe_smape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        denominator[denominator == 0] = 1e-8
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    mape = safe_mape(actual, forecast)
    smape = safe_smape(actual, forecast)
    masked_mape = safe_mape(actual[actual > 50], forecast[actual > 50])

    st.subheader("Evaluation Metrics")
    st.markdown(f"""
    - **RMSE**: `{rmse:.2f}`  
    - **MAE**: `{mae:.2f}`  
    - **MAPE**: `{mape:.2f}%`  
    - **SMAPE**: `{smape:.2f}%`  
    - **Masked MAPE (PAC > 50)**: `{masked_mape:.2f}%`
    """)
else:
    st.warning("Not enough test data available for forecasting.")
