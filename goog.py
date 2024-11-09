#google stock price prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import ta

df = pd.read_csv("GOOG.csv")

#feature engineering
df["SMA_200"] = df["close"].rolling(window=200).mean()

#computing the stats to see rsi in price
delta = df["close"].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss


df["RSI"] = 100 - (100 / (1 + rs))
df["EMA_12"] = df["close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["close"].ewm(span=26, adjust=False).mean()

df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
df["macd"] = df["EMA_50"] - df["EMA_200"]
df["MACD_200"] = df["macd"].ewm(span=9, adjust=False).mean()

df.dropna(inplace=True)

X = df[["low", "open", "close", "volume", "SMA_200", "RSI", "EMA_12", "EMA_26", "EMA_200", "MACD", "MACD_signal", "macd", "MACD_200"]]
Y = df[["high"]]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


model = GradientBoostingRegressor(learning_rate=0.0003, n_estimators=5000, criterion="friedman_mse", random_state=0)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

#strong performance
#print(f"Training Score: {model.score(X_train, Y_train)}") 0.94
#print(f"Testing Score: {model.score(X_test, Y_test)}") 0.93
#print(f"Training MSE: {mean_squared_error(Y_train, y_pred_train)}") 7560
#print(f"Testing MSE: {mean_squared_error(Y_test, y_pred)}") 4875


