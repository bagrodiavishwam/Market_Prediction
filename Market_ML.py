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
del snsx["Stock Splits"]
