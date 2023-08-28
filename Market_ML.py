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

def backtest(data, model, predictors, start=2520, step=252):
    '''start=2520 is that we are taking 2520 dataentries, which is equivalent to 10 years
    step=252 means that we will take 252 entries and then proceed to the next year and repeat the same
    essentially what we will do is take the values of the first 10 years and predict the value of the 11th year
    then take the values of the first 11 years and predict the value for the 12th year and so on'''

    all_predictions=[]
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test=data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

predictions = backtest(snsx, model, predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

predictions["Target"].value_counts() / predictions.shape[0]

horizons=[2,5,60,252,1004]
new_predictors=[]


for horizon in horizons:
    rolling_averages=snsx.rolling(horizon).mean()
    
    ratio_column=f"CLose_Ratio_{horizon}"
    snsx[ratio_column]=snsx["Close"]/rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    snsx[trend_column]= snsx.shift(1).rolling(horizon).sum()["Target"]

    new_predictors+= [ratio_column, trend_column]
    
snsx=snsx.dropna().copy()

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions=backtest(snsx,model, new_predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

