import yfinance as yf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

nyse = 'symbols/nyse_symbols.csv'
nasdaq = 'symbols/nasdaq_symbols.csv'


"""
BASIC INDICATORS HELPER FUNCTIONS
"""

SMOOTHING_FACTOR = 2

def getHistory(symbol, period, interval):
    ticker = yf.Ticker(symbol)
    historical = ticker.history(period=period, interval=interval, actions=False)
    return historical


def getMACD(historical, ret=False):
    shortEMA = getEMA(historical, 12)
    longEMA = getEMA(historical, 26)
    if shortEMA is None or longEMA is None:
        #TODO: THROW ERROR!!!
        return None
    n = longEMA.shape[0]
    shortEMA = shortEMA[-n:]
    MACD = shortEMA - longEMA
    signal = getEMA(MACD, 9)
    if signal is None:
        #TODO: THROW ERROR!!!
        return None
    n = signal.shape[0]
    MACD = MACD[-n:]
    if ret:
        return MACD, signal, shortEMA, longEMA
    return MACD, signal, None, None


def getAVG(data):
    return np.sum(data) / data.shape[0]


def getEXP(curr_val, EMA_old, n):
    smoothing = SMOOTHING_FACTOR / (1 + n)
    return (curr_val * smoothing) + (EMA_old * (1 - smoothing))


def getEMA(data, n):
    if data.shape[0] < n:
        #Throw an error
        return None
    ema = np.zeros((data.shape[0] - n) + 1)
    ema[0] = getAVG(data[:n])      # Initial EMA value is a normal average of however many periods
    for i in range(data.shape[0] - n):
        ema[i+1] = getEXP(data[i+n], ema[i], n)
    return ema


def getMA(data, n):
    if data.shape[0] < n:
        #TODO: throw an error
        return None
    ma = np.zeros((data.shape[0] - n) + 1)
    for i in range((data.shape[0] - n) + 1):
        ma[i] = getAVG(data[i:i+n])
    return ma


def getVWAP(close, high, low, volume):
    n = close.shape[0]
    vwap = np.zeros(n)
    total_vp = 0
    total_volume = 0
    for i in range(n, 0, -1):
        total_vp += ((close[-i] + high[-i] + low[-i])/3) * volume[-i]
        total_volume += volume[-i]
        vwap[-i] = total_vp / total_volume
    return vwap


"""
STRATEGIES HELPER FUNCTIONS
"""

#STRATEGIES
UPTREND_BASE = 3
EMA_PERIODS = 200

def strategyMACD(symbol, n):
    """
    If the MACD line crosses above the signal line, below 0, on the latest
    candlestick, return True
    Otherwise return False
    """
    try:
        close = getHistory(symbol, '1mo', '30m')['Close'].to_numpy()[:-n]
        macd, signal, _, _ = getMACD(close)

        # Check that macd has been below the signal line
        for i in range(5, 1, -1):
            if macd[-i] > 0 and signal[-i] > 0:
                return False
            if macd[-i] > signal[-i]:
                return False
        if macd[-1] < signal[-1]:
            return False
        return True
    except:
        return False


def strategyEMA(symbol, n):
    """
    If stock closing price is below the EMA at any point in the last 3 candle
    sticks, then return False
    return True if it has been in an uptrend for at least 3 candlesticks
    """
    try:
        close = getHistory(symbol, '1mo', '30m')['Close'].to_numpy()[:-n]
        ema = getEMA(close, EMA_PERIODS)

        for i in range(5, 0, -1):
            if close[-i] < ema[-i]:
                return False
        return True
    except:
        return False


def strategyMA(symbol, n):
    try:
        close = getHistory(symbol, '1mo', '5m')['Close'].to_numpy()[:-n*6]
        ma = getMA(close, 200)
        for i in range(5, 0, -1):
            if close[-i] < ma[-i]:
                return False
        return True
    except:
        return False


def strategyVWAP(symbol, n):
    try:
        historical = getHistory(symbol, '1mo', '5m')
        close = hitorical['Close'].to_numpy()[:-n*6]
        high = historical['High'].to_numpy()[:-n*6]
        low = historical['Low'].to_numpy()[:-n*6]
        volume = historical['Volume'].to_numpy()[:-n*6]
        vwap = getVWAP(close, high, low, volume)

        # check that the stock price is above both the vwap and the ma
        for i in range(5, 0, -1):
            if close[-i] < vwap[-i]:
                return False
        return True
    except:
        return False


"""
GENERAL USE FUNCTIONS
"""

def stopLosses(buyPrice, risk, n):
    L = []
    L.append(buyPrice * (1-risk))
    for i in range(1,n):
        L.append(buyPrice * (1+(risk*i)))
    return L


def sellSignals(buyPrice, risk, reward):
    stop1 = buyPrice * (1 - risk)
    stop2 = buyPrice * (1 + risk)
    stop3 = buyPrice * (1 + reward)
    return stop1, stop2, stop3


def howManyShares(MoneyAvailable, buyPrice):
    return MoneyAvailable // buyPrice


def buySignals(strategies, symbol, n):
    for strategy in strategies:
        if not strategy(symbol, n):
            return False
    return True


def scanStocks(symbols, strategies, n=0):
    buy = np.zeros(len(symbols))
    for i in range(len(symbols)):
        if buySignals(strategies, symbols[i], n):
            buy[i] = 1
    return buy


def filterStocks(symbols, minVolume=300000, minClose=1, maxClose=100):
    filtered = []
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).history(period='1mo', interval='30m', actions=False)
            if info['Close'][-1] > 1 and info['Close'][-1] < 100 and sum(info['Volume'][-10:])/10 > minVolume:
                filtered.append(symbol)
        except:
            pass
    return filtered


def readSymbols(files=[nyse, nasdaq]):
    symbols = []
    for file in files:
        with open(file, 'r') as f:
            for line in f.readlines():
                symbols.append(line.strip().split(',')[0])
    return symbols


MACD = strategyMACD
EMA = strategyEMA
VWAP = strategyVWAP
MA = strategyMA

if __name__ == "__main__":
    stock = getHistory("AMD", "1mo", "30m")
    close = stock["Close"].to_numpy()

    # test getting otheer shit
    macd, signal, _, _ = getMACD(close)
    ema200 = getEMA(close, 200)
    t1 = time.time()
    symbols = readSymbols()[:200]
    print(len(symbols))

    buyResult = scanStocks(symbols, [MACD, EMA], '1mo', '30m')
    print(len(buyResult))

    buyIdx = np.nonzero(buyResult == 1)[0]
    for i in buyIdx:
        print(symbols[i])
    t2 = time.time()
    print(t2-t1)
    print(len(buyIdx))
    # buy(buyResult)
#    x = np.array(range(ema200.shape[0]))
#    y0 = np.zeros(ema200.shape[0])
#
#    close = close[-ema200.shape[0]:]
#    macd = macd[-ema200.shape[0]:]
#    signal = signal[-ema200.shape[0]:]
#
#    fig, (plt1, plt2) = plt.subplots(2, figsize=(10, 5))
#    plt2.plot(x, macd, color="cyan")   # macd line
#    plt2.plot(x, signal, color="orange")  # signal line
#    plt2.plot(x, y0, color="black")
#
#    plt1.plot(x, close, color="blue")   # 12 day EMA
#    plt1.plot(x, ema200, color="red")    # 26 day EMA
#    plt1.plot(x, close, color="black")
#
#    plt.show()

