import pandas as pd
import numpy as np

BB_PERIOD      = 20      
BB_STD         = 2.0     
Z_ENTRY        = -2.0    
Z_EXIT         = 0.0     
RSI_PERIOD     = 14
RSI_OVERSOLD   = 30      
RSI_EXIT       = 55      
ATR_PERIOD     = 14
ATR_STOP_MULT  = 2.0     
ATR_TP_MULT    = 3.0    

DATA_PATH   = "Nifty_50_Historical_Data.csv"
OUT_SIGNALS = "signals_nifty_50.csv"

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    def clean_num(col):
        return (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Close"] = clean_num("Price")
    df["Open"]  = clean_num("Open")
    df["High"]  = clean_num("High")
    df["Low"]   = clean_num("Low")

    df = df[["Date", "Open", "High", "Low", "Close"]].sort_values("Date").reset_index(drop=True)
    return df

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr

def compute_indicators(df):
    df = df.copy()
    sma = df["Close"].rolling(BB_PERIOD).mean()
    std = df["Close"].rolling(BB_PERIOD).std()

    df["BB_Mid"]   = sma
    df["BB_Upper"] = sma + BB_STD * std
    df["BB_Lower"] = sma - BB_STD * std
    df["ZScore"]   = (df["Close"] - sma) / std

    df["RSI"] = compute_rsi(df["Close"], RSI_PERIOD)
    df["ATR"] = compute_atr(df["High"], df["Low"], df["Close"], ATR_PERIOD)
    return df

def generate_signals(df):
    df = df.copy()
    n = len(df)

    position = np.zeros(n, dtype=int)   
    signal   = np.zeros(n, dtype=int)  

    in_position = False
    entry_price = None
    stop_price = None
    tp_price = None

    close   = df["Close"].values
    zscore  = df["ZScore"].values
    rsi     = df["RSI"].values
    atr     = df["ATR"].values

    for i in range(n):
        if np.isnan(zscore[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]):
            position[i] = 1 if in_position else 0
            continue

        if not in_position:
            entry_condition = (
                zscore[i] <= Z_ENTRY and
                rsi[i] <= RSI_OVERSOLD
            )
            if entry_condition:
                in_position = True
                entry_price = close[i]
                stop_price = entry_price - ATR_STOP_MULT * atr[i]
                tp_price = entry_price + ATR_TP_MULT * atr[i]
                signal[i] = 1
                position[i] = 1
            else:
                signal[i] = 0
                position[i] = 0
        else:
            exit_condition = (
                zscore[i] >= Z_EXIT or
                rsi[i] >= RSI_EXIT or
                close[i] <= stop_price or
                close[i] >= tp_price
            )
            if exit_condition:
                in_position = False
                signal[i] = -1
                position[i] = 0
                entry_price = None
                stop_price = None
                tp_price = None
            else:
                signal[i] = 0
                position[i] = 1

    df["Position"] = position
    df["Signal"] = signal
    return df

def main():
    df = load_data(DATA_PATH)
    df = compute_indicators(df)
    df = generate_signals(df)

    out = df[["Date", "Close", "Position", "Signal"]].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT_SIGNALS, index=False)

    print(f"Processed {len(out)} rows from {out['Date'].iloc[0]} to {out['Date'].iloc[-1]}")
    print(f"Buy signals: {(out['Signal']==1).sum()}  Sell signals: {(out['Signal']==-1).sum()}")
    print(f"Saved: {OUT_SIGNALS}")

if __name__ == "__main__":
    main()
