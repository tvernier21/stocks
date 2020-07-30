import numpy as np
import yfinance as yf
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import tkinter as tk
import os

import helper

BUY = 'buy_file.csv'

nyse = 'symbols/nyse_symbols.csv'
nasdaq = 'symbols/nasdaq_symbols.csv'

RISK = .01
REWARD = .02

START_MONEY = 10000
CAP_MONEY = 1000
CAP_TRADES = 10

def graphStock(symbol, i):
    close = helper.getHistory(symbol, '1mo', '30m')['Close'].to_numpy()
    macd, signal, _, _ = helper.getMACD(close)
    ema200 = helper.getEMA(close, 200)
    n = ema200.shape[0]

    macd = macd[-n:]
    signal = signal[-n:]
    close = close[-n:]

    x = np.array(range(n))
    y0 = np.zeros(n)

    fig, (plt1, plt2, plt3) = plt.subplots(3, figsize=(8, 8))
    plt2.plot(x, macd, color="cyan")   # macd line
    plt2.plot(x, signal, color="orange")  # signal line
    plt2.plot(x, y0, color="black")

    plt1.plot(x, close, color="blue")   # 12 day EMA
    plt1.plot(x, ema200, color="red")    # 26 day EMA

    stops = helper.sellSignals(close[0], RISK, REWARD)

    for stop in stops:
        plt1.plot(x, np.zeros(n)+stop, color='green')


    stock = helper.getHistory(symbol, '1mo', '5m')
    vwap = helper.getVWAP(stock['Close'].to_numpy(),
                          stock['High'].to_numpy(),
                          stock['Low'].to_numpy(),
                          stock['Volume'].to_numpy(),
                          stock.index)
    ma = helper.getMA(stock.Close, 200)
    x = np.array(range(ma.shape[0]))
    vwap = vwap[-ma.shape[0]:]
    closer = stock.Close.to_numpy()[-ma.shape[0]:]
    plt3.plot(x, vwap)
    plt3.plot(x, closer)
    plt3.plot(x, ma)


    plt.show()


def save(buy, money, new=False):
    buy_str = ['','','']
    if not new:
        buy_str = read()

    buy_str[0] += ',' + str(money)
    for symbol in buy:
        buy_str[1] += ',' + symbol
        buy_str[2] += ',' + str(helper.getHistory(symbol, '1mo', '30m')['Close'].to_numpy()[-1])

    with open(BUY, 'w') as f:
        for line in buy_str:
            f.write(line+'\n')
    f.close()


def read():
    buy_str = []
    try:
        with open(BUY, 'r') as f:
            for line in f.readlines():
                buy_str.append(line.strip())
    except:
        pass
    return buy_str


def runFilter(files, strategies, interval, period, RISK, REWARD):
    symbols = helper.readSymbols(files)
    scanned = helper.scanStocks(symbols, strategies, interval, period)
    remove_idx = np.nonzero(scanned == -1)[0]
    buy_idx = np.nonzero(scanned == 1)[0]

    buy = [symbols[i] for i in buy_idx]       # put all buy stocks into list
    offset = 0
    for i in remove_idx:
        symbols.pop(i-offset)
        offset+=1
    save(buy, symbols, START_MONEY)


def parseFiles(string):
    return string.strip().split(' ')


def getStrategies(macd, ema):
    l=[]
    if macd:
        l.append(helper.MACD)
    if ema:
        l.append(helper.EMA)
    return l


def save_symbols(symbols, param):
    with open(SYMBOLS, 'w') as f:
        f.write(str(param[0])+','+str(param[1])+','+str(param[2])+','+str(param[3])+'\n')
        for symbol in symbols:
            f.write(symbol+'\n')


def save_main(money, symbols, init_price, stop_loss, new=False):
    main_str = ['','','','']       # [money, symbols, init_price, stop_loss]

    for i in money:
        main_str[0] += ',' + str(i)
    for i in range(len(symbols)):
        main_str[1] += ',' + str(symbol[i])
        main_str[2] += ',' + str(init_price[i])
        main_str[3] += ',' + str(stop_loss[i])

    if new:
        main_str[0] = main_str[0][1:]
    with open(BUY, 'w') as f:
        for line in main_str:
            f.write(line+'\n')


def Today():
    files = [nyse]
    t1 = time.time()
    symbols = helper.readSymbols(files)
    risk = .01
    reward = .02
    strats = [helper.MACD, helper.EMA]
    stopwatch = 0

    # clean up
    offset = 0
    for i in range(len(symbols)):
        try:
            sumfin = helper.getHistory(symbols[i-offset], '1mo', '30m')['Close'].to_numpy()[-1]
        except:
            print(f'popped {symbols[i-offset]}')
            symbols.pop(i-offset)
            offset += 1

    bought = []
    price_bought = []
    time_bought = []
    bottom = []
    sold = []
    price_sold0 = []
    price_sold1 = []
    time_sold0 = []
    time_sold1 = []

    t2 = time.time()
    loop = t2 - t1
    while True:
        print('Scanning All Stocks')
        t1 = time.time()
        scanned = helper.scanStocks(symbols, strats, '1mo', '30m')
        buy_idx = np.nonzero(scanned == 1)[0]
        print(buy_idx)
        for i in buy_idx:
            print(symbols[i])
            if symbols[i] not in bought:
                bought.append(symbols[i])
                price_bought.append(helper.getHistory(symbols[i], '1mo','30m')['Close'].to_numpy()[-1])
                bgak = datetime.datetime.now()
                time_bought.append(f'{bgak.hour}:{bgak.minute}')
                bottom.append(0)
        t2 = time.time()
        timetoscan = t2-t1
        print(timetoscan)
        loop += timetoscan

        while loop < 1800:
            t1 = time.time()
            offset = 0
            for i in range(len(bought)):
                try:
                    curr_price = helper.getHistory(bought[i-offset], '1mo', '30m')['Close'].to_numpy()[-1]
                except:
                    continue
                signals = helper.sellSignals(price_bought[i-offset], .01, .05)
                if curr_price < signals[bottom[i-offset]]:
                    sold.append(bought.pop(i-offset))
                    price_sold0.append(price_bought.pop(i-offset))
                    price_sold1.append(curr_price)
                    time_sold0.append(time_bought.pop(i-offset))
                    bgak = datetime.datetime.now()
                    time_sold1.append(f'{bgak.hour}:{bgak.minute}')
                    bottom.pop(i-offset)
                    offset += 1
                elif curr_price >= signals[bottom[i-offset]+1]:
                    bottom[i-offset] += 1
                    if curr_price >= signals[bottom[i-offset]+1]:
                        sold.append(bought.pop(i-offset))
                        price_sold0.append(price_bought.pop(i-offset))
                        price_sold1.append(curr_price)
                        time_sold0.append(time_bought.pop(i-offset))
                        bgak = datetime.datetime.now()
                        time_sold1.append(f'{bgak.hour}:{bgak.minute}')
                        bottom.pop(i-offset)
                        offset += 1

            #print info
            print('\n\n\n')
            print('-- BUY --')
            for i in range(len(bought)):
                print(f'{bought[i]} ({price_bought[i]})   @ {time_bought[i]}')
            print('\n-- SELL --')
            for i in range(len(sold)):
                print(f'{sold[i]} B({price_sold0[i]}) @ {time_sold0[i]} S({price_sold1[i]}) @ {time_sold1[i]}')
            print('\n\n\n')
            t2 = time.time()
            timez = t2 - t1
            time.sleep(300 - timez)
            loop_time= time.time() - t1
            loop += loop_time
            if loop > 1500:
                time.sleep(1800-loop)
                loop += (time.time() - t2)
        loop = 0



def grapher(stock, signals):
    n = stock.shape[0]
    x = np.array(range(n))
    fig, (plt1) = plt.subplots(1, figsize=(6,6))
    plt1.plot(x, stock, color='black')
    for s in signals:
        plt1.plot(x, np.zeros(n)+s, color='green')

    plt.show()


def calculateProfit(buyPrice, sellPrice, numStocks):
    diff = sellPrice - buyPrice
    return diff*numStocks


def backTest():
    files = [nyse]
    symbols = helper.readSymbols(files)
    risk = .01
    reward = .02
    # strats = [helper.MACD, helper.EMA, helper.VWAP, helper.MA]
    strats = [helper.VWAP, helper.MA]
    # strats = [helper.MACD]

    # Removed old stocks
    print(len(symbols))
    symbols = helper.filterStocks(symbols, minClose=.03, maxClose=500, minVolume=10000)    #default settings left alone
    print(len(symbols))

    #Scan each stock and check for win signal
    account = np.array([0])
    trades = 0
    success = 0
    total_profit = 0
    bought = np.zeros(70)     # tracking number of trades invested at each timestamp
    for symbol in symbols:
        # graphStock(symbol, 0)
        profit = 0
        for i in range(65, 0, -1):   # looking backwards 5 days
            if helper.buySignals(strats, symbol, i):
                print('Buy')
                bought[-i] += 1
                # Look at the stocks closing prices in 5 min intervals from the close candle right after where we get the signal
                close = helper.getHistory(symbol, '1mo', '5m')['Close'].to_numpy()[-i*6:]
                numShares = helper.howManyShares(500, close[0])
                stopLoss = helper.stopLosses(close[0], .01, 10)

                bottom = 0
                trades += 1
                for idx in range(close.shape[0]):
                    if close[idx] < stopLoss[bottom]:
                        break
                    elif close[idx] >= stopLoss[bottom + 1]:
                        bottom += 1
                        if close[idx] >= stopLoss[9]:
                            break
                profit += calculateProfit(close[0], stopLoss[bottom], numShares)
                if bottom > 0:
                    success += 1
        total_profit += profit
        print(f'{symbol} made {profit}')
        account = np.append(account, total_profit)

    print(total_profit)
    print(success, '/', trades)


    print(bought)
    n=account.shape[0]
    x = np.array(range(n))
    plt.plot(x, account)
    plt.show()



if __name__ == "__main__":
    # Today()
    backTest()


"""
root = tk.Tk()

canvas = tk.Canvas(root, height=800, width=1000, bg='#263D42')
canvas.pack()

filterFrame = tk.Frame(root, bg='#4a7782')
filterFrame.place(relx=.05, rely=.01, relwidth=.9, relheight=.2)

symbolsFile = tk.Entry(filterFrame, font=40)
symbolsFile.insert(0, 'symbols/nyse_symbols.csv symbols/nasdaq_symbols.csv')
symbolsFile.place(relwidth=.3, relheight=.2, relx=.01, rely=.1)

risk = tk.Entry(filterFrame, font=40)
risk.insert(0, '.01')
risk.place(relwidth=.3, relheight=.2, relx=.33, rely=.1)

reward = tk.Entry(filterFrame, font=40)
reward.insert(0, '.02')
reward.place(relwidth=.3, relheight=.2, relx=.65, rely=.1)

macd = tk.IntVar()
macd_button = tk.Checkbutton(filterFrame, text='MACD', variable=macd)
macd_button.place(relwidth=.3, relheight=.2, relx=.01, rely=.4)

ema = tk.IntVar()
ema_button = tk.Checkbutton(filterFrame, text='EMA', variable=ema)
ema_button.place(relwidth=.3, relheight=.2, relx=.33, rely=.4)

scanButton = tk.Button(filterFrame, text='Start Scan',
       command=lambda: startScan(symbolsFile.get(), risk.get(), reward.get(), macd.get(), ema.get()))
scanButton.place(relwidth=.3, relheight=.3, relx=.65, rely=.5)



stockFrame = tk.Frame(root, bg='#4a7782')
stockFrame.place(relx=.05, rely=.23, relwidth=.3, relheight=.75)

profitFrame = tk.Frame(root, bg='#4a7782')
profitFrame.place(relx=.35, rely=.23, relwidth=.65, relheight=.75)


while True:
    scanAll()
    while timer < 1800:
        scanBuy()
        updateTable()
        updateGraph()
        root.update()

root.mainloop()


"""

