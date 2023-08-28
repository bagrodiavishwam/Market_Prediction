import yfinance as yf
import pandas as pd
import os

if os.path.exists("snsx.csv"):
    snsx = pd.read_csv("snsx.csv", index_col=0)
else:
    snsx = yf.Ticker("^NSEI")
    snsx = snsx.history(period="max")
    snsx.to_csv("snsx.csv")

del snsx ["Dividends"] 
del snsx["Stock Splits

snsx["Tomorrow"] = snsx["Close"].shift(-1)

snsx["Target"] = (snsx["Tomorrow"] > snsx["Close"]).astype(int)
# snsx = snsx.loc["1990-01-01":].copy()

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = snsx.iloc[:-100]
test = snsx.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)


combined = pd.concat([test["Target"], preds], axis=1)
combined.columns=["Target", "Prediction"]
combined.plot()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    # combined
    return combined
