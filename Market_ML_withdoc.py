```python
# Importing necessary libraries
import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Checking if data file "snsx.csv" exists, otherwise fetching data from Yahoo Finance
if os.path.exists("snsx.csv"):
    snsx = pd.read_csv("snsx.csv", index_col=0)
else:
    snsx = yf.Ticker("^NSEI")
    snsx = snsx.history(period="max")
    snsx.to_csv("snsx.csv")

# Data preprocessing
del snsx["Dividends"]
del snsx["Stock Splits"]
snsx["Tomorrow"] = snsx["Close"].shift(-1)
snsx["Target"] = (snsx["Tomorrow"] > snsx["Close"]).astype(int)

# Defining predictor columns
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Splitting data into train and test sets
train = snsx.iloc[:-100]
test = snsx.iloc[-100:]

# Creating and training the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(train[predictors], train["Target"])

# Making predictions and calculating precision score
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision = precision_score(test["Target"], preds)

# Displaying combined target and prediction plot
combined = pd.concat([test["Target"], preds], axis=1)
combined.columns = ["Target", "Prediction"]
combined.plot()

# Defining a function to predict using the trained model
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Defining a function to perform backtesting
def backtest(data, model, predictors, start=2520, step=252):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

# Performing backtesting with different horizons
predictions = backtest(snsx, model, predictors)
predictions["Predictions"].value_counts()

# Precision score for the backtested predictions
precision_backtest = precision_score(predictions["Target"], predictions["Predictions"])

# Displaying precision scores for different horizons
predictions["Target"].value_counts() / predictions.shape[0]

# Defining different horizons for feature engineering
horizons = [2, 5, 60, 252, 1004]
new_predictors = []

# Feature engineering and creating new predictor columns
for horizon in horizons:
    rolling_averages = snsx.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    snsx[ratio_column] = snsx["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    snsx[trend_column] = snsx.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

# Dropping rows with missing values after feature engineering
snsx = snsx.dropna().copy()

# Creating a new Random Forest Classifier model after feature engineering
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Redefining the predict function to utilize predicted probabilities and threshold
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds_proba = model.predict_proba(test[predictors])[:, 1]
    preds = (preds_proba >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Performing backtesting with the new model and predictors
predictions = backtest(snsx, model, new_predictors)

# Displaying value counts of predicted classes
predictions["Predictions"].value_counts()

# Precision score for the backtested predictions with new model and predictors
precision_backtest_new = precision_score(predictions["Target"], predictions["Predictions"])
```

This code involves the use of the `yfinance` library to fetch stock market data, data preprocessing, feature engineering, training a Random Forest Classifier, performing backtesting with different horizons, and evaluating precision scores for different scenarios. The code is aimed at predictive modeling for stock price trends using machine learning techniques.
