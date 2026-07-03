"""
data_fetcher.py
Fetches stock price history and recent news for a given ticker.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_data(ticker: str, days: int = 90) -> dict:
    """
    Fetch OHLCV data + key stats for a ticker.
    Returns a dict with summary text chunks ready for embedding.
    """
    end   = datetime.today()
    start = end - timedelta(days=days)

    stock = yf.Ticker(ticker)
    hist  = stock.history(start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True)

    if hist.empty:
        return {"ticker": ticker, "chunks": [], "error": "No data found"}

    close   = hist["Close"]
    volume  = hist["Volume"]
    returns = close.pct_change().dropna()

    # ── Key metrics ────────────────────────────────────────────────────────────
    current_price  = close.iloc[-1]
    price_1w_ago   = close.iloc[-6]  if len(close) > 5  else close.iloc[0]
    price_1m_ago   = close.iloc[-22] if len(close) > 21 else close.iloc[0]
    price_3m_ago   = close.iloc[0]

    ret_1w  = (current_price - price_1w_ago)  / price_1w_ago  * 100
    ret_1m  = (current_price - price_1m_ago)  / price_1m_ago  * 100
    ret_3m  = (current_price - price_3m_ago)  / price_3m_ago  * 100
    vol_ann = returns.std() * (252 ** 0.5) * 100

    ma20  = close.rolling(20).mean().iloc[-1]
    ma50  = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    avg_vol = volume.mean()

    # ── RSI ────────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = (100 - (100 / (1 + gain / loss))).iloc[-1]

    # ── Trend detection ────────────────────────────────────────────────────────
    trend = "uptrend" if current_price > ma20 else "downtrend"
    rsi_signal = (
        "overbought (RSI > 70)" if rsi > 70
        else "oversold (RSI < 30)" if rsi < 30
        else "neutral"
    )

    # ── Top gainers / losers days ──────────────────────────────────────────────
    best_day  = returns.idxmax()
    worst_day = returns.idxmin()

    # ── Build natural-language chunks ─────────────────────────────────────────
    chunks = [
        f"{ticker} current price as of {end.strftime('%Y-%m-%d')} is ${current_price:.2f}.",

        f"{ticker} performance: 1-week return {ret_1w:+.2f}%, "
        f"1-month return {ret_1m:+.2f}%, 3-month return {ret_3m:+.2f}%.",

        f"{ticker} technical indicators: 20-day moving average is ${ma20:.2f}"
        + (f", 50-day moving average is ${ma50:.2f}" if ma50 else "")
        + f". The stock is currently in a {trend} relative to its 20-day MA.",

        f"{ticker} RSI is {rsi:.1f}, which is {rsi_signal}.",

        f"{ticker} annualised volatility over the past {days} days is {vol_ann:.1f}%.",

        f"{ticker} average daily trading volume is {avg_vol:,.0f} shares.",

        f"{ticker} best single day in the period: {best_day.strftime('%Y-%m-%d')} "
        f"(+{returns[best_day]*100:.2f}%). "
        f"Worst day: {worst_day.strftime('%Y-%m-%d')} ({returns[worst_day]*100:.2f}%).",
    ]

    # ── Fundamental info ───────────────────────────────────────────────────────
    try:
        info = stock.info
        pe   = info.get("trailingPE")
        mktc = info.get("marketCap")
        sect = info.get("sector")
        name = info.get("longName", ticker)

        if pe:
            chunks.append(f"{ticker} ({name}) trailing P/E ratio is {pe:.1f}.")
        if mktc:
            chunks.append(f"{ticker} market cap is ${mktc/1e9:.1f}B.")
        if sect:
            chunks.append(f"{ticker} operates in the {sect} sector.")
    except Exception:
        pass

    return {"ticker": ticker, "chunks": chunks, "error": None}


def fetch_portfolio_data(tickers: list[str]) -> list[dict]:
    """Fetch data for all tickers in the portfolio."""
    results = []
    for ticker in tickers:
        print(f"  📥 Fetching {ticker}...")
        result = fetch_stock_data(ticker)
        if not result["error"]:
            results.append(result)
        else:
            print(f"  ⚠️  {ticker}: {result['error']}")
    return results
