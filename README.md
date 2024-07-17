# Instagram Reach Analysis and Forecasting

This repository contains a comprehensive analysis and forecasting of Instagram reach data. The tasks include data preprocessing, visualizations, statistical analysis, and time series forecasting using the SARIMA model.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Analysis Steps](#analysis-steps)
  - [1. Import Libraries and Load Data](#1-import-libraries-and-load-data)
  - [2. Check for Null Values, Column Info, and Descriptive Statistics](#2-check-for-null-values-column-info-and-descriptive-statistics)
  - [3. Convert Date Column to Datetime](#3-convert-date-column-to-datetime)
  - [4. Analyze the Trend of Instagram Reach Over Time](#4-analyze-the-trend-of-instagram-reach-over-time)
  - [5. Analyze Instagram Reach for Each Day](#5-analyze-instagram-reach-for-each-day)
  - [6. Analyze the Distribution of Instagram Reach](#6-analyze-the-distribution-of-instagram-reach)
  - [7. Calculate Mean, Median, and Standard Deviation of Instagram Reach for Each Day](#7-calculate-mean-median-and-standard-deviation-of-instagram-reach-for-each-day)
  - [8. Visualize Reach for Each Day of the Week](#8-visualize-reach-for-each-day-of-the-week)
  - [9. Check Trends and Seasonal Patterns](#9-check-trends-and-seasonal-patterns)
  - [10. Forecast Instagram Reach Using SARIMA Model](#10-forecast-instagram-reach-using-sarima-model)
- [Contributing](#contributing)


## Introduction

This project aims to analyze and forecast Instagram reach using historical data. By understanding the patterns and trends in the data, we can develop a predictive model to forecast future reach, helping content creators optimize their posting strategy.

## Project Overview

In this project, we perform the following steps:
1. Data Preprocessing: Cleaning and preparing the data for analysis.
2. Exploratory Data Analysis (EDA): Visualizing data to understand trends, patterns, and distributions.
3. Statistical Analysis: Calculating key statistics for different days of the week.
4. Time Series Decomposition: Identifying and analyzing seasonal and trend components.
5. Forecasting: Using the SARIMA model to predict future Instagram reach and evaluating model performance.

## Analysis Steps

### 1. Import Libraries and Load Data

Load necessary libraries and the dataset for analysis.

```python

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = 'path_to_your_file/Instagram-Reach (1).csv'
data = pd.read_csv(file_path)

# Display first few rows of the dataset
data.head()
```

### 2. Check for Null Values, Column Info, and Descriptive Statistics

Check the dataset for any null values and understand its structure and basic statistics.

```python
# Check for null values
null_values = data.isnull().sum()
print(null_values)

# Display column info
column_info = data.info()

# Display descriptive statistics
descriptive_stats = data.describe()
print(descriptive_stats)
```
### 3. Convert Date Column to Datetime

Convert the 'Date' column to a datetime format for time series analysis.

```python
# Convert the Date column into datetime datatype
data['Date'] = pd.to_datetime(data['Date'])
data.head()
```
### 4. Analyze the Trend of Instagram Reach Over Time

Visualize the trend of Instagram reach using a line chart.

```python
# Analyze the trend of Instagram reach over time using a line chart
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Instagram reach'])
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.title('Trend of Instagram Reach Over Time')
plt.show()

```
### 5. Analyze Instagram Reach for Each Day

Visualize Instagram reach for each day of the week using a bar chart.

```python
# Create a 'Day' column
data['Day'] = data['Date'].dt.day_name()

# Analyze Instagram reach for each day using a bar chart
plt.figure(figsize=(12, 6))
data.groupby('Day')['Instagram reach'].sum().plot(kind='bar')
plt.xlabel('Day')
plt.ylabel('Total Instagram Reach')
plt.title('Instagram Reach for Each Day')
plt.show()
```
### 6. Analyze the Distribution of Instagram Reach

Use a box plot to understand the distribution and identify outliers.

```python
# Analyze the distribution of Instagram reach using a box plot
plt.figure(figsize=(12, 6))
data.boxplot(column='Instagram reach')
plt.ylabel('Instagram Reach')
plt.title('Distribution of Instagram Reach')
plt.show()
```
### 7. Calculate Mean, Median, and Standard Deviation of Instagram Reach for Each Day

Group the data by day and calculate statistical metrics.

```python
# Group the DataFrame by the Day column and calculate the mean, median, and standard deviation of the Instagram reach for each day
grouped_data = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
print(grouped_data)
```

### 8. Visualize Reach for Each Day of the Week

Create a bar chart to visualize average reach for each day of the week.

```python
# Create a bar chart to visualize the reach for each day of the week
plt.figure(figsize=(12, 6))
data.groupby('Day')['Instagram reach'].mean().plot(kind='bar')
plt.xlabel('Day')
plt.ylabel('Mean Instagram Reach')
plt.title('Average Instagram Reach for Each Day of the Week')
plt.show()
```
### 9. Check Trends and Seasonal Patterns

Perform seasonal decomposition to identify trends and seasonal patterns.

```python
# Decompose the time series data
data.set_index('Date', inplace=True)
result = seasonal_decompose(data['Instagram reach'], model='additive', period=30)
result.plot()
plt.show()
```
### 10. Forecast Instagram Reach Using SARIMA Model

Fit a SARIMA model to the data and make predictions. Evaluate the model using MAE and MSE.

```python
# Step 10: Forecast Instagram Reach Using SARIMA Model

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = 'path_to_your_file/Instagram-Reach (1).csv'
data = pd.read_csv(file_path)

# Convert the Date column into datetime datatype
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Ensure the datetime index has a frequency set
data = data.asfreq('D')

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plot_acf(data['Instagram reach'], lags=30)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data['Instagram reach'], lags=30)
plt.show()

# Split the data into train and test sets
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# Fit the SARIMA model
model = SARIMAX(train['Instagram reach'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecast
forecast = results.get_forecast(steps=len(test))
forecast_df = forecast.summary_frame()

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Instagram reach'], label='Train')
plt.plot(test.index, test['Instagram reach'], label='Test')
plt.plot(forecast_df.index, forecast_df['mean'], label='Forecast')
plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='k', alpha=0.1)
plt.xlabel('Date')
plt.ylabel('Instagram Reach')
plt.title('Instagram Reach Forecast')
plt.legend()
plt.show()

# Calculate error metrics
mae = mean_absolute_error(test['Instagram reach'], forecast_df['mean'])
mse = mean_squared_error(test['Instagram reach'], forecast_df['mean'])
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

```
## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements.



