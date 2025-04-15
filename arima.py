import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv(r"C:\Users\Manas\Downloads\monthly_sales.csv", parse_dates=["Month"])
df.set_index("Month", inplace=True)

# Plot original data
plt.figure(figsize=(10, 4))
plt.plot(df, marker='o')
plt.title("Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# Stationarity check with ADF test
result = adfuller(df['Sales'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] <= 0.05:
    print(" Series is stationary")
else:
    print("Series is not stationary, differencing needed")

# Differencing if needed
df_diff = df.diff().dropna()

# Plot differenced data
plt.figure(figsize=(10, 4))
plt.plot(df_diff, marker='o', color='orange')
plt.title("Differenced Sales Data")
plt.xlabel("Date")
plt.ylabel("Sales Change")
plt.grid(True)
plt.show()

# Fit ARIMA model (manual order for simplicity: ARIMA(1,1,1))
model = ARIMA(df['Sales'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Forecast next 5 steps
forecast = model_fit.forecast(steps=5)
print("\nForecast for next 5 months:")
print(forecast)

# Plot forecast
plt.figure(figsize=(10, 4))
plt.plot(df, label="Historical")
plt.plot(pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=5, freq='M'),
         forecast, label="Forecast", marker='o', color='red')
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
