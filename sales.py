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

print("skew is", store.SALES.skew()) #it gives 1.16
print("kurtosis is" store.SALES.kurt()) # it gives 1.79

num=store.select_dtypes(include=[np.number])
print(num.dtypes)

corr=num.corr()
print(corr)

f, ax =plt.subplots(figsize=(12,9))
sns.heatmap(corr, vmax=1, square=True)

sns.countplot(store.STATUS)

YEAR_ID_pivot= \
store.pivot_table(index="YEAR_ID", values="SALES", aggfunc=np.median)
YEAR_ID_pivot.plot(kind="bar", color="pink", figsize=(12,7))
plt.xlabel("year")
plt.ylabel("sales")
plt.title("sales per year")
plt.xticks(rotation=0)
plt.show()

