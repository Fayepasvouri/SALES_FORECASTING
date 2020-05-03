"""
faye
"""
# Importing required libraries
import numpy as np
import pandas as pd, datetime
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
from time import time
import os
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
import xgboost as xgb
import warnings

store = pd.read_csv(
    "C:/Users/Faye/Downloads/sales_data_sample.csv", encoding="ISO-8859-1"
)
store.head()

plt.style.use("fivethirtyeight")
plt.figure(figsize=(12,7))
sns.distplot(store.SALES, bins=25)
plt.ticklabel_format(style="plain", axis="x", scilimits=(0,1))
plt.xlabel("Sales")
plt.ylabel("Number_of_sales")
plt.title("sales_distribution")
