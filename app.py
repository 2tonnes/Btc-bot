import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from flask import Flask
from metaapi_cloud_sdk import MetaApi

app = Flask(__name__)

METAAPI_TOKEN = os.environ.get('METAAPI_TOKEN')
ACCOUNT_ID = os.environ.get('ACCOUNT_ID')

SYMBOL = 'BTCUSDm'
TIMEFRAME = '15m'
FORCE_MIN_LOT = True
MIN_LOT = 0.01
RISK_PCT = 0.01
MAGIC = 990011

LOOKBACK = 10
RSI_LEN = 2
RSI_ENTRY_LOW = 5.0
RSI_ENTRY_HIGH = 97.0
RSI_EXIT_HIGH = 70.0
RSI_EXIT_LOW = 40.0
TREND_LEN = 200
ATR_LEN = 14
ATR_STOP_MULT = 3.0
MAX_BARS = 96
USE_TREND_FILTER = True

ADD_TRIGGER_ATR = 1.0
ADD_RISK_FRACTION = 0.5
TRAIL_TRIGGER_ATR = 1.5
TRAIL_DISTANCE_ATR = 1.0
MAX_ADDS = 2


async def get_account_and_connection():
    api = MetaApi(METAAPI_TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    if account.state != 'DEPLOYED':
        await account.deploy()
    await account.wait_connected()
    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()
    return account, connection


async def fetch_candles(account, count=250):
    candles = await account.get_historical_candles(SYMBOL, TIMEFRAME, datetime.now(timezone.utc), count)
    return candles


def candles_to_df(candles):
    df = pd.DataFrame(candles)
    df = df.sort_values('time').reset_index(drop=True)
    return df[['time', 'open', 'high', 'low', 'close']]


def generate_signal(df):
    df = df.copy()
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_LEN).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(RSI_LEN).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_LEN).mean()
    df['rsi2'] = 100 - (100 / (1 + gain / loss))

    df['sma200'] = df['close'].rolling(TREND_LEN).mean()
    df['prior_high'] = df['high'].rolling(LOOKBACK).max().shift(1)
    df['prior_low'] = df['low'].rolling(LOOKBACK).min().shift(1)
    df['swept_low'] = (df['low'] < df['prior_low']) & (df['close'] > df['prior_low'])
    df['swept_high'] = (df['high'] > df['prior_high']) & (df['close'] < df['prior_high'])

    trend_ok_long = (not USE_TREND_FILTER) or (df['close'] > df['sma200'])
    trend_ok_short = (not USE_TREND_FILTER) or (df['close'] < df['sma200'])
    df['long_signal'] = df['swept_low'] & (df['rsi2'] <= RSI_ENTRY_LOW) & trend_ok_long
    df['short_signal'] = df['swept_high'] & (df['rsi2'] >= RSI_ENTRY_HIGH) & trend_ok_short
    return df


def calc_lot_size(balance, risk_pct, stop_distance, pip_value_per_lot=1):
    risk_amount = balance * risk_pct
    lots = risk_amount / (stop_distance * pip_value_per_lot)
    return round(max(lots, MIN_LOT), 2)


async def get_our_positions(connection):
    positions = await connection.get_positions()
    return [p for p in positions if p['symbol'] == SYMBOL and p.get('magic') == MAGIC]


async def manage_open_positions(connection, positions, latest):
    direction = 1 if positions[0]['type'] == 'POSITION_TYPE_BUY' else -1
    price = latest['close']
    atr = latest['atr']
    entry_price = positions[0]['openPrice']
    move_favorable = (price - entry_price) if direction == 1 else (entry_price - price)

    if move_favorable >= TRAIL_TRIGGER_ATR * atr:
        new_stop = (price - TRAIL_DISTANCE_ATR * atr) if direction == 1 else (price + TRAIL_DISTANCE_ATR * atr)
        for pos in positions:
            current_stop = pos.get('stopLoss')
            better = (current_stop is None) or \
                     (direction == 1 and new_stop > current_stop) or \
                     (direction == -1 and new_stop < current_stop)
            if better:
                print(f"Trailing stop update on position {pos['id']} to {round(new_stop,2)}")
                await connection.modify_position(pos['id'], stop_loss=new_stop)

    if len(positions) - 1 < MAX_ADDS and move_favorable >= ADD_TRIGGER_ATR * atr * len(positions):
        stop_dist = ATR_STOP_MULT * atr
        add_lots = MIN_LOT if FORCE_MIN_LOT else calc_lot_size(positions[0]['equity'], RISK_PCT * ADD_RISK_FRACTION, stop_dist)
        stop_price = price - stop_dist if direction == 1 else price + stop_dist
        print(f"Adding to position: {add_lots} lots, direction={direction}")
        if direction == 1:
            await connection.create_market_buy_order(SYMBOL, add_lots, stop_loss=stop_price, options={'magic': MAGIC})
        else:
            await connection.create_market_sell_order(SYMBOL, add_lots, stop_loss=stop_price, options={'magic': MAGIC})

    rsi_exit = (latest['rsi2'] >= RSI_EXIT_HIGH) if direction == 1 else (latest['rsi2'] <= RSI_EXIT_LOW)
    if rsi_exit:
        print("RSI exit condition met. Closing all positions.")
        for pos in positions:
            await connection.close_position(pos['id'])


async def open_new_position(connection, direction, latest, balance):
    stop_dist = ATR_STOP_MULT * latest['atr']
    lots = MIN_LOT if FORCE_MIN_LOT else calc_lot_size(balance, RISK_PCT, stop_dist)
    stop_price = latest['close'] - stop_dist if direction == 1 else latest['close'] + stop_dist
    print(f"Opening {'LONG' if direction==1 else 'SHORT'}: {lots} lots, stop={round(stop_price,2)}")
    if direction == 1:
        result = await connection.create_market_buy_order(SYMBOL, lots, stop_loss=stop_price, options={'magic': MAGIC})
    else:
        result = await connection.create_market_sell_order(SYMBOL, lots, stop_loss=stop_price, options={'magic': MAGIC})
    print(f"Order result: {result}")


async def run_cycle():
    account, connection = await get_account_and_connection()
    account_info = await connection.get_account_information()
    balance = account_info['balance']

    candles = await fetch_candles(account)
    df = candles_to_df(candles)
    df = generate_signal(df)
    latest = df.iloc[-1]

    log = []
    log.append(f"Balance: {balance}")
    log.append(f"Close: {latest['close']}, RSI2: {round(latest['rsi2'],2)}, ATR: {round(latest['atr'],2)}")
    log.append(f"Long signal: {latest['long_signal']} | Short signal: {latest['short_signal']}")

    positions = await get_our_positions(connection)
    log.append(f"Open positions (ours): {len(positions)}")

    if positions:
        await manage_open_positions(connection, positions, latest)
    elif latest['long_signal']:
        await open_new_position(connection, 1, latest, balance)
    elif latest['short_signal']:
        await open_new_position(connection, -1, latest, balance)
    else:
        log.append("No signal, no open position. No action.")

    for line in log:
        print(line)
    return "OK"


@app.route('/run')
def trigger():
    result = asyncio.run(run_cycle())
    return result, 200


@app.route('/')
def home():
    return "Bot is alive", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
