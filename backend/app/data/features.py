# backend/app/data/features.py
import numpy as np
import pandas as pd

def EMA(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def MACD(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist

def ATR(df: pd.DataFrame, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.fillna(method='bfill')

def VWAP(df: pd.DataFrame):
    tp = (df['high'] + df['low'] + df['close']) / 3
    vwap = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def price_action_features(df: pd.DataFrame):
    body = (df['close'] - df['open']).abs()
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    last5_trend = df['close'].pct_change().rolling(5).sum()
    return pd.DataFrame({
        'body': body,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'last5_trend': last5_trend
    }).fillna(0)

def create_features(df: pd.DataFrame):
    """
    Input df columns: ['open','high','low','close','volume','timestamp']
    Returns df with engineered features (no scaling).
    """
    df = df.copy().reset_index(drop=True)
    df['rsi'] = RSI(df['close'])
    df['ema9'] = EMA(df['close'], 9)
    df['ema21'] = EMA(df['close'], 21)
    df['ema50'] = EMA(df['close'], 50)
    macd, macd_sig, macd_hist = MACD(df['close'])
    df['macd'] = macd
    df['macd_sig'] = macd_sig
    df['macd_hist'] = macd_hist
    df['atr'] = ATR(df)
    df['vwap'] = VWAP(df)
    pa = price_action_features(df)
    df = pd.concat([df, pa], axis=1)
    # forward fill or fillna
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df