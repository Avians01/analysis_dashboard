# pv data analysis for PV1
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from tabulate import tabulate

DATA_PATH = "D:/truxco energy dataset"
df = pd.read_csv(os.path.join(DATA_PATH, 'pv-data.csv'), low_memory=False)

print(df.head())
data_types = dict(df.dtypes)
print(data_types)
print("\n")

# Converting datetime_millis into datetime
df['datetime'] = pd.to_datetime(df['datetime_millis'], errors='coerce')
print("Datetime parsing sample:")
print(df['datetime'].head(5))
print("\n")

# Dropping rows with missing datetime or pac values
df = df.dropna(subset=['datetime', 'pac'])
print(f"Original rows: {len(df)}")
df = df.dropna(subset=['datetime', 'pac'])
print(f"Rows after dropping invalid datetime/power: {len(df)}")
print("\n")

# datetime as index
df = df.set_index('datetime')
df = df.sort_index()

print("Index type:", type(df.index))
print("First and last timestamps:")
print("Datetime range:", df.index.min(), "to", df.index.max())
print("Is datetime sorted?", df.index.is_monotonic_increasing)
print("\n")

#focussing on PV001
df_device = df[df['device_id'] == 'PV-001']

print("PAC Stats:")
print(df_device['pac'].describe())

print("\nEAC Today Stats:")
print(df_device['eac_today'].describe())

print("\nEAC Total Stats:")
print(df_device['eac_total'].describe())
print("\n")

print("Unique device IDs in original data:", df['device_id'].unique())
df_device = df[df['device_id'] == 'PV-001']
print("Shape after filtering PV-001:", df_device.shape)
print("\n")

print("Power stats for PV-001:")
print(df_device['pac'].describe())
print("Any non-zero values?", (df_device['pac'] != 0).any())

df_device['pac'].iloc[:1000].plot(title="PV-001 PAC Sample Plot")
plt.show()

df_device['pac'].plot(figsize=(12, 4), title="PV-001 pac Output – First View")
plt.show()

df_hourly = df_device['pac'].resample('h').mean()
df_hourly.plot(title="pac - PV001- hourly plot")
plt.show()

df_device = df[df['device_id'] == 'PV-001']
df_hourly = df_device['pac'].resample('h').mean().reset_index()  # Reset index to get datetime as a column

# Plot using plotly express
fig = px.line(df_hourly, x='datetime', y='pac',
              title='Hourly Mean PAC Output - PV001',
              labels={'pac': 'Power (W)', 'datetime': 'Time'})

fig.update_layout(xaxis_title='Time', yaxis_title='Average PAC (W)',
                  template='plotly_white', hovermode='x unified')

fig.show()

#plot inverter_status over time
df_hourly_status = df_device[['inverter_status']].resample('h').mean().reset_index()

fig = px.line(df_hourly_status, x='datetime', y='inverter_status',
              title='Hourly Mean Inverter Status - PV001',
              labels={'inverter_status': 'Inverter Status', 'datetime': 'Time'})

fig.show()

#TIME SERIES ANALYSIS
if 'datetime' not in df.columns:
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

df_device = df[df['device_id'] == 'PV-001'].copy()
df_device = df_device.set_index('datetime')
df_device_numeric = df_device.select_dtypes(include='number')
df_hourly = df_device_numeric.resample('h').mean()
df_hourly = df_hourly.interpolate()

#seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_hourly['pac'], model='additive', period=24)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.suptitle("Seasonal Decomposition of PAC – PV-001", fontsize=16)
result.plot()
plt.tight_layout()
plt.show()

# EDA WITH SEABORN
df_device = df_device.reset_index()

#Temperature Distribution
sns.histplot(df_device['temp1'], kde=True)
plt.title("Temperature Distribution – PV-001")
plt.xlabel("Temperature Sensor 1")
plt.ylabel("Frequency")
plt.show()

#Power vs. Temperature
sns.scatterplot(x='temp1', y='real_output_power', data=df_device)
plt.title("Power Output vs Temperature – PV-001")
plt.xlabel("Temperature (temp1)")
plt.ylabel("Real Output Power (W)")
plt.show()

#Diurnal Pattern: Boxplot of Output Power by Hour
df_device['hour'] = df_device['datetime'].dt.hour
sns.boxplot(x='hour', y='real_output_power', data=df_device)
plt.title("Hourly Distribution of Power Output – PV-001")
plt.xlabel("Hour of Day")
plt.ylabel("Real Output Power (W)")
plt.show()

#Augmented Dickey-Fuller (ADF)
from statsmodels.tsa.stattools import adfuller
print("\n")
adf_result = adfuller(df_hourly['pac'].dropna())
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("\n")

# Autocorrelation Plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df_hourly['pac'].dropna(), lags=50)
plt.title("ACF of PAC")
plt.show()

plot_pacf(df_hourly['pac'].dropna(), lags=50)
plt.title("PACF of PAC")
plt.show()



# Correlation Analysis 
cols = ['pac', 'inverter_status', 'temp1', 'temp2', 'temp3']
sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix with PAC")
# plt.show()

# Granger causality test
from statsmodels.tsa.stattools import grangercausalitytests
df = df.sort_values(by='datetime_millis')
df = df[['pac', 'inverter_status']].dropna()
grangercausalitytests(df, maxlag=4)

from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np

#log transform to pac
df_hourly['pac_log'] = np.log1p(df_hourly['pac'])  # log(1 + pac) to handle 0 safely
df_hourly.columns = df_hourly.columns.str.strip()
#exog var
exog_vars = ['temp2']
target_log = df_hourly['pac_log']

# print(type(df_hourly))               
# print(df_hourly.columns)            
# print('pac' in df_hourly.columns)   
# print(df_hourly.columns.tolist())

#Train-Test split using the log-transformed target
train_log = target_log[:'2025-05-29']
test_log = target_log['2025-05-30':]

# Actual PAC values (untransformed) for comparison
actual = df_hourly['pac']['2025-05-30':]

print(actual.describe())

print((actual < 100).sum(), "/", len(actual), "values under 100")

# extract exogenous variables aligned with train/test
exog_train = df_hourly.loc[train_log.index, exog_vars]
exog_test = df_hourly.loc[test_log.index, exog_vars]

# print(df_hourly.head())
# print(df_hourly.index)
# print(df_hourly.dtypes)

# SARIMAX model with exogenous variables on log-transformed target
model = SARIMAX(train_log, 
                exog=exog_train,
                order=(2, 1, 2), 
                seasonal_order=(1, 1, 1, 24),
                enforce_stationarity=False, 
                enforce_invertibility=False)

results = model.fit(disp=False)
print(results.summary())
print("\n")

# Forecasting in log-space
forecast_log = results.predict(start=test_log.index[0], end=test_log.index[-1], exog=exog_test, dynamic=False)

# Inverse transform to original scale
forecast = np.expm1(forecast_log)

# Plot Forecast vs Actual (original PAC scale)
plt.figure(figsize=(14, 5))
plt.plot(np.expm1(train_log[-7*24:]), label='Train')
plt.plot(actual, label='Actual')
plt.plot(forecast, label='Forecast')
plt.title("SARIMA Forecast vs Actual (PV Output)")
plt.xlabel("Date")
plt.ylabel("Power Output (pac)")
plt.grid(True)
plt.legend()
plt.show()

# Residuals on original scale
residuals = actual - forecast
plt.figure()
plt.plot(residuals)
plt.title("Forecast Residuals")
plt.xlabel("Date")
plt.ylabel("Error")
plt.grid(True)
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

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

print(f"SARIMA RMSE : {rmse:.2f}")
print(f"SARIMA MAE  : {mae:.2f}")
print(f"SARIMA MAPE : {mape:.2f}%")
print(f"SARIMA SMAPE: {smape:.2f}%")

# Masked MAPE: exclude near-zero values
mask = actual > 50
masked_mape = safe_mape(actual[mask], forecast[mask])
print("MAPE (masked):", masked_mape)


# Plot Residual ACF/PACF
plot_acf(residuals.dropna(), lags=50)
plt.title("ACF of Residuals")
plt.show()

plot_pacf(residuals.dropna(), lags=50)
plt.title("PACF of Residuals")
plt.show()

