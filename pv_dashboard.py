# pv data analysis for PV1
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from tabulate import tabulate
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.units as munits
import pandas.plotting as pd_plotting

# Clear previous unit converters to avoid UnitData bugs
munits.registry.clear()

# Register pandas converters properly
pd_plotting.register_matplotlib_converters()


# --- SETTINGS ---
plt.rcParams["figure.figsize"] = (14, 6)  # Default larger figsize for clarity
sns.set_style("whitegrid")

# --- LOAD DATA ---
DATA_PATH = "D:/truxco energy dataset"
df = pd.read_csv(os.path.join(DATA_PATH, 'pv-data.csv'), low_memory=False)
df['datetime'] = pd.to_datetime(df['datetime_millis'], errors='coerce')
df = df.dropna(subset=['datetime', 'pac'])

df = df.set_index('datetime')
df = df.sort_index()
df_device = df[df['device_id'] == 'PV-001']

# --- PLOTTING PAC OVERVIEW ---
df_device['pac'].iloc[:1000].plot(title="PV-001 PAC Sample Plot")
plt.tight_layout()
plt.show()

df_device['pac'].plot(title="PV-001 PAC Output – Full Series")
plt.tight_layout()
plt.show()

# --- HOURLY PAC PLOT ---
df_hourly = df_device['pac'].resample('h').mean()

df_hourly.plot(title="Hourly Mean PAC - PV001")
plt.tight_layout()
plt.show()

# --- PLOTLY INTERACTIVE PLOT ---
df_hourly_df = df_hourly.reset_index()
fig = px.line(df_hourly_df, x='datetime', y='pac',
              title='Hourly Mean PAC Output - PV001',
              labels={'pac': 'Power (W)', 'datetime': 'Time'})
fig.update_layout(xaxis_title='Time', yaxis_title='Average PAC (W)', template='plotly_white', hovermode='x unified')
fig.show()

# --- INVERTER STATUS OVER TIME ---
df_status = df_device[['inverter_status']].resample('h').mean().reset_index()
fig = px.line(df_status, x='datetime', y='inverter_status',
              title='Hourly Mean Inverter Status - PV001',
              labels={'inverter_status': 'Inverter Status', 'datetime': 'Time'})
fig.show()

# --- SEASONAL DECOMPOSITION ---
df_device_numeric = df_device.select_dtypes(include='number')
df_hourly_all = df_device_numeric.resample('h').mean().interpolate()
decomp_result = seasonal_decompose(df_hourly_all['pac'], model='additive', period=24)

plt.figure(figsize=(14, 10))
decomp_result.plot()
plt.suptitle("Seasonal Decomposition of PAC – PV-001", fontsize=16)
plt.tight_layout()
plt.show()

# --- EDA: TEMPERATURE + OUTPUT ---
df_device_reset = df_device.reset_index()

sns.histplot(df_device_reset['temp1'], kde=True)
plt.title("Temperature Distribution – PV-001")
plt.xlabel("Temperature Sensor 1")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

sns.scatterplot(x='temp1', y='real_output_power', data=df_device_reset)
plt.title("Power Output vs Temperature – PV-001")
plt.xlabel("Temperature (temp1)")
plt.ylabel("Real Output Power (W)")
plt.tight_layout()
plt.show()

df_device_reset['hour'] = df_device_reset['datetime'].dt.hour
sns.boxplot(x='hour', y='real_output_power', data=df_device_reset)
plt.title("Hourly Distribution of Power Output – PV-001")
plt.xlabel("Hour of Day")
plt.ylabel("Real Output Power (W)")
plt.tight_layout()
plt.show()

# --- STATIONARITY CHECK ---
print("\nADF Test:")
adf_result = adfuller(df_hourly_all['pac'].dropna())
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value      : {adf_result[1]:.4f}\n")

# --- ACF/PACF ---
plot_acf(df_hourly_all['pac'].dropna(), lags=50)
plt.title("ACF of PAC")
plt.tight_layout()
plt.show()

plot_pacf(df_hourly_all['pac'].dropna(), lags=50)
plt.title("PACF of PAC")
plt.tight_layout()
plt.show()

# --- CORRELATION HEATMAP ---
corr_cols = ['pac', 'inverter_status', 'temp1', 'temp2', 'temp3']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix with PAC")
plt.tight_layout()
plt.show()

# --- GRANGER CAUSALITY ---
print("\nGranger Causality Test (inverter_status → pac):")
granger_df = df[['pac', 'inverter_status']].dropna()
grangercausalitytests(granger_df, maxlag=4)

# --- SARIMA FORECASTING ---
df_hourly_all['pac_log'] = np.log1p(df_hourly_all['pac'])
exog_vars = ['temp2']
target_log = df_hourly_all['pac_log']

train_log = target_log[:'2025-05-29']
test_log = target_log['2025-05-30':]
actual = df_hourly_all['pac']['2025-05-30':]
exog_train = df_hourly_all.loc[train_log.index, exog_vars]
exog_test = df_hourly_all.loc[test_log.index, exog_vars]

model = SARIMAX(train_log, 
                exog=exog_train,
                order=(2, 1, 2), 
                seasonal_order=(1, 1, 1, 24),
                enforce_stationarity=False, 
                enforce_invertibility=False)

results = model.fit(disp=False)

# --- CLEAN SUMMARY PRINTING ---
summary_txt = results.summary().as_text()
print("\n=== SARIMA MODEL SUMMARY ===")
print(summary_txt)
print("="*40 + "\n")

# Optionally save summary to .txt
with open("sarima_model_summary.txt", "w") as f:
    f.write(summary_txt)

# --- FORECASTING ---
forecast_log = results.predict(start=test_log.index[0], end=test_log.index[-1], exog=exog_test, dynamic=False)
forecast = np.expm1(forecast_log)

plt.figure(figsize=(14, 6))
plt.plot(np.expm1(train_log[-7*24:]), label='Train')
plt.plot(actual, label='Actual')
plt.plot(forecast, label='Forecast')
plt.title("SARIMA Forecast vs Actual (PV Output)")
plt.xlabel("Date")
plt.ylabel("Power Output (pac)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- RESIDUALS ---
residuals = actual - forecast
plt.plot(residuals)
plt.title("Forecast Residuals")
plt.xlabel("Date")
plt.ylabel("Error")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- ERROR METRICS ---
rmse = np.sqrt(mean_squared_error(actual, forecast))
mae = mean_absolute_error(actual, forecast)

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def safe_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator[denominator == 0] = 1e-8
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)

mape = safe_mape(actual, forecast)
smape = safe_smape(actual, forecast)
masked_mape = safe_mape(actual[actual > 50], forecast[actual > 50])

print(f"SARIMA RMSE : {rmse:.2f}")
print(f"SARIMA MAE  : {mae:.2f}")
print(f"SARIMA MAPE : {mape:.2f}%")
print(f"SARIMA SMAPE: {smape:.2f}%")
print(f"MAPE (masked, >50W): {masked_mape:.2f}%")

# --- RESIDUAL DIAGNOSTICS ---
plot_acf(residuals.dropna(), lags=50)
plt.title("ACF of Forecast Residuals")
plt.tight_layout()
plt.show()

plot_pacf(residuals.dropna(), lags=50)
plt.title("PACF of Forecast Residuals")
plt.tight_layout()
plt.show()
