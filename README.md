This program uses historical data to predict the next-day trend in the Indian market.

Currently, the program uses the SENSEX index or the S&P BSE Index.

Certainly! Let's break down the code step by step to understand what each section does:

1. **Importing Libraries:**
   ```python
   import yfinance as yf
   import pandas as pd
   import os
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import precision_score
   ```
   This section imports the necessary libraries for data manipulation, fetching stock data, machine learning, and evaluation metrics.

2. **Checking and Loading Data:**
   ```python
   if os.path.exists("snsx.csv"):
       snsx = pd.read_csv("snsx.csv", index_col=0)
   else:
       snsx = yf.Ticker("^NSEI")
       snsx = snsx.history(period="max")
       snsx.to_csv("snsx.csv")
   ```
   The code checks if a data file named "snsx.csv" exists. If it does, the data is loaded from the CSV file. If not, the code fetches historical stock data for the Nifty 50 index from Yahoo Finance using the `yfinance` library, saves it to the CSV file, and loads it into a Pandas DataFrame named `snsx`.

3. **Data Preprocessing:**
   ```python
   del snsx["Dividends"]
   del snsx["Stock Splits"]
   snsx["Tomorrow"] = snsx["Close"].shift(-1)
   snsx["Target"] = (snsx["Tomorrow"] > snsx["Close"]).astype(int)
   ```
   This section removes unnecessary columns, creates a "Tomorrow" column with the next day's closing price, and calculates the "Target" column by comparing tomorrow's closing price with today's closing price to determine whether the price will increase (1) or not (0).

4. **Defining Predictor Columns and Train-Test Split:**
   ```python
   predictors = ["Close", "Volume", "Open", "High", "Low"]
   train = snsx.iloc[:-100]
   test = snsx.iloc[-100:]
   ```
   The `predictors` list holds the column names used as features for training the model. The data is split into training and testing sets, with the last 100 rows used for testing.

5. **Training a Random Forest Classifier:**
   ```python
   model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
   model.fit(train[predictors], train["Target"])
   ```
   A Random Forest Classifier model is initialized with specified parameters and trained on the training data using the specified predictor columns.

6. **Making Predictions and Calculating Precision:**
   ```python
   preds = model.predict(test[predictors])
   preds = pd.Series(preds, index=test.index)
   precision = precision_score(test["Target"], preds)
   ```
   The model makes predictions on the test data using the predictor columns. The predictions are then stored in a Pandas Series. The precision score is calculated by comparing the predicted values with the actual target values.

7. **Displaying Target and Prediction Plot:**
   ```python
   combined = pd.concat([test["Target"], preds], axis=1)
   combined.columns = ["Target", "Prediction"]
   combined.plot()
   ```
   The code combines the actual target values and predicted values into a DataFrame named `combined` and creates a plot showing the comparison between the target and prediction.

8. **Defining Predict and Backtest Functions:**
   ```python
   def predict(train, test, predictors, model):
       model.fit(train[predictors], train["Target"])
       preds = model.predict(test[predictors])
       preds = pd.Series(preds, index=test.index, name="Predictions")
       combined = pd.concat([test["Target"], preds], axis=1)
       return combined

   def backtest(data, model, predictors, start=2520, step=252):
       # ...
   ```
   The `predict` function takes training and test data, predictor columns, and a model as input, fits the model to the training data, predicts target values for the test data, and returns a DataFrame containing the actual target values and predictions.

   The `backtest` function performs a backtesting simulation by iterating through the data in steps, training the model on historical data, and making predictions for a future window of time. This is repeated with different windows to simulate how the model would perform over time.

9. **Backtesting and Evaluating Precision:**
   ```python
   predictions = backtest(snsx, model, predictors)
   precision_backtest = precision_score(predictions["Target"], predictions["Predictions"])
   ```
   The `backtest` function is called to perform backtesting on the `snsx` data using the specified model and predictor columns. The precision score for the backtested predictions is calculated by comparing the target and predicted values.

10. **Feature Engineering:**
   ```python
   horizons = [2, 5, 60, 252, 1004]
   new_predictors = []
   # ...
   ```
   Different horizons are defined for feature engineering. Rolling averages and trend columns are calculated and added to the DataFrame `snsx` as new predictor columns.

11. **Re-defining Prediction Function and Backtesting:**
   ```python
   model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
   def predict(train, test, predictors, model):
       # ...
   predictions = backtest(snsx, model, new_predictors)
   ```
   A new Random Forest Classifier model is created with updated parameters. The `predict` function is redefined to predict probabilities, apply a threshold, and convert predicted probabilities into classes (0 or 1).

12. **Evaluating Backtested Predictions with New Features:**
   ```python
   precision_backtest_new = precision_score(predictions["Target"], predictions["Predictions"])
   ```
   The `backtest` function is applied again with the new model and predictors. The precision score for the backtested predictions with the new features is calculated and stored.

Overall, this code demonstrates a series of steps for fetching stock market data, training machine learning models, making predictions, performing backtesting with different horizons, and evaluating the precision of predictions. Feature engineering is also applied to enhance the predictive power of the model.
