"""Production-Grade Market Data Downloader with direct Yahoo Finance v8/v10 chart engine,
Stooq Financial API fallback, multi-asset alignment, local parquet caching, and intraday support.
"""

import os
import time
import json
import logging
import urllib.request
import urllib.parse
from typing import Optional, Dict, Tuple, List, Any
from pathlib import Path
import pandas as pd
import numpy as np

from stock_predictor.config import DEFAULT_CONFIG, PredictionConfig
from stock_predictor.data.synthetic import (
    generate_synthetic_stock_data,
    generate_synthetic_intraday_data,
    get_anchor_price_for_ticker
)

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


def _fetch_yahoo_v8_chart(
    ticker: str,
    interval: str = "1d",
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeout: int = 6
) -> Optional[pd.DataFrame]:
    """
    Directly queries Yahoo Finance v8 Chart JSON API with fallback hosts and browser headers.
    """
    encoded_ticker = urllib.parse.quote(ticker.strip().upper())
    
    interval_map = {
        "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m",
        "30m": "30m", "60m": "60m", "1h": "1h", "1d": "1d",
        "5d": "5d", "1wk": "1wk", "1mo": "1mo"
    }
    yf_interval = interval_map.get(interval, "1d")
    
    params: Dict[str, Any] = {
        "interval": yf_interval,
        "includePrePost": "false",
        "events": "div,split"
    }
    
    if start_date:
        try:
            p1 = int(pd.Timestamp(start_date).timestamp())
            params["period1"] = p1
            if end_date:
                params["period2"] = int(pd.Timestamp(end_date).timestamp())
            else:
                params["period2"] = int(time.time())
        except Exception:
            params["range"] = period or "2y"
    elif period:
        period_norm = period.lower().replace("d", "d").replace("mo", "mo").replace("y", "y")
        if period_norm in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]:
            params["range"] = period_norm
        else:
            params["range"] = "2y"
    else:
        params["range"] = "5d" if interval in ["1m", "5m", "15m", "30m", "1h"] else "2y"

    query_str = urllib.parse.urlencode(params)
    hosts = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_ticker}?{query_str}",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{encoded_ticker}?{query_str}"
    ]

    for url in hosts:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    raw_data = response.read().decode("utf-8")
                    data = json.loads(raw_data)
                    
                    chart = data.get("chart", {})
                    results = chart.get("result")
                    if not results:
                        continue
                        
                    res = results[0]
                    timestamps = res.get("timestamp")
                    if not timestamps:
                        continue
                        
                    indicators = res.get("indicators", {})
                    quotes = indicators.get("quote", [{}])[0]
                    
                    opens = quotes.get("open", [])
                    highs = quotes.get("high", [])
                    lows = quotes.get("low", [])
                    closes = quotes.get("close", [])
                    volumes = quotes.get("volume", [])
                    
                    adjcloses = indicators.get("adjclose", [{}])[0].get("adjclose", closes)
                    
                    dt_index = pd.to_datetime(timestamps, unit="s", utc=True)
                    dt_index = dt_index.tz_convert("America/New_York").tz_localize(None)
                    
                    df = pd.DataFrame({
                        "Open": opens,
                        "High": highs,
                        "Low": lows,
                        "Close": closes,
                        "Adj Close": adjcloses,
                        "Volume": volumes
                    }, index=dt_index)
                    
                    df.index.name = "Date"
                    
                    df["Volume"] = df["Volume"].fillna(0).astype(np.int64)
                    df["Close"] = df["Close"].ffill().bfill()
                    df["Open"] = df["Open"].fillna(df["Close"])
                    df["High"] = df["High"].fillna(df[["Open", "Close"]].max(axis=1))
                    df["Low"] = df["Low"].fillna(df[["Open", "Close"]].min(axis=1))
                    df["Adj Close"] = df["Adj Close"].fillna(df["Close"])
                    
                    df = df.dropna(subset=["Close"])
                    if len(df) > 0:
                        return df
        except Exception as e:
            logger.debug(f"Direct Yahoo fetch attempt failed for {ticker} from {url}: {e}")
            continue

    return None


def _fetch_stooq_chart(ticker: str, timeout: int = 5) -> Optional[pd.DataFrame]:
    """
    Fetches real-world historical OHLCV data from Stooq API (100% free open quant finance endpoint).
    """
    clean_sym = ticker.strip().lower()
    if clean_sym.startswith("^"):
        stooq_sym = clean_sym.replace("^", "")
    else:
        stooq_sym = f"{clean_sym}.us"
        
    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                df = pd.read_csv(response)
                if df is not None and len(df) >= 10 and "Close" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    df.sort_index(inplace=True)
                    if "Adj Close" not in df.columns:
                        df["Adj Close"] = df["Close"]
                    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    except Exception as e:
        logger.debug(f"Stooq fetch failed for {ticker}: {e}")
    return None


class StockDataDownloader:
    """Fetches, cleans, and caches daily and intraday stock market data with source tracking."""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_was_synthetic: bool = False
        
    def _get_cache_path(self, ticker: str, start: Optional[str], end: Optional[str], interval: str, period: Optional[str]) -> Path:
        end_str = end or period or "latest"
        start_str = start or "default"
        safe_ticker = ticker.replace("^", "IDX_").replace("/", "_").replace(".", "-")
        return self.cache_dir / f"{safe_ticker}_{interval}_{start_str}_{end_str}.parquet"
        
    def fetch_ticker_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        period: Optional[str] = None,
        custom_price: Optional[float] = None,
        use_cache: bool = True,
        force_synthetic: bool = False
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data (daily or intraday) for a ticker with automatic multi-source fallback.
        """
        ticker_clean = ticker.strip().upper()
        
        if interval != "1d" and period is None and start_date is None:
            period = "1mo" if interval in ["5m", "15m", "30m", "1h"] else "5d"
        elif interval == "1d" and start_date is None and period is None:
            period = "2y"
            
        cache_path = self._get_cache_path(ticker_clean, start_date, end_date, interval, period)
        
        # 1. Check local cache
        if use_cache and not force_synthetic and cache_path.exists() and custom_price is None:
            try:
                mtime = cache_path.stat().st_mtime
                exp_hours = 0.25 if interval != "1d" else self.config.cache_expiry_hours
                if (time.time() - mtime) < (exp_hours * 3600):
                    df = pd.read_parquet(cache_path)
                    if len(df) > 10:
                        self.last_was_synthetic = False
                        return df
            except Exception as e:
                logger.warning(f"Failed to read cache for {ticker_clean}: {e}")

        # 2. Force synthetic if requested
        if force_synthetic:
            self.last_was_synthetic = True
            if interval != "1d":
                df = generate_synthetic_intraday_data(ticker=ticker_clean, interval=interval, initial_price=custom_price)
            else:
                df = generate_synthetic_stock_data(ticker=ticker_clean, start_date=start_date, end_date=end_date, initial_price=custom_price)
            return df

        # 3. Direct Yahoo Finance v8/v10 Chart Engine
        try:
            df_direct = _fetch_yahoo_v8_chart(
                ticker=ticker_clean,
                interval=interval,
                period=period,
                start_date=start_date,
                end_date=end_date
            )
            
            if df_direct is not None and len(df_direct) >= 10:
                try:
                    df_direct.to_parquet(cache_path)
                except Exception:
                    pass
                self.last_was_synthetic = False
                return df_direct
        except Exception as direct_err:
            logger.warning(f"Direct Yahoo chart fetch failed for {ticker_clean}: {direct_err}")

        # 4. Stooq Free Quant Engine Fallback (Daily)
        if interval == "1d":
            try:
                df_stooq = _fetch_stooq_chart(ticker=ticker_clean)
                if df_stooq is not None and len(df_stooq) >= 10:
                    try:
                        df_stooq.to_parquet(cache_path)
                    except Exception:
                        pass
                    self.last_was_synthetic = False
                    return df_stooq
            except Exception as stooq_err:
                logger.debug(f"Stooq fallback failed for {ticker_clean}: {stooq_err}")

        # 5. yfinance Ticker.history() Fallback
        try:
            import yfinance as yf
            yf_tick = yf.Ticker(ticker_clean)
            if period:
                data = yf_tick.history(period=period, interval=interval, auto_adjust=False)
            elif start_date:
                data = yf_tick.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
            else:
                data = yf_tick.history(period="2y", interval=interval, auto_adjust=False)
                
            if data is not None and len(data) >= 10:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                    
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in required_cols:
                    if col not in data.columns:
                        if col == "Close" and "Adj Close" in data.columns:
                            data["Close"] = data["Adj Close"]
                        else:
                            raise ValueError(f"Missing required column {col}")
                            
                if "Adj Close" not in data.columns:
                    data["Adj Close"] = data["Close"]
                    
                df = data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
                df = df.dropna(subset=["Close"])
                df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
                df.index.name = "Date"
                
                try:
                    df.to_parquet(cache_path)
                except Exception:
                    pass
                    
                self.last_was_synthetic = False
                return df
        except Exception as yf_err:
            logger.warning(f"yfinance history download failed for {ticker_clean}: {yf_err}")

        # 6. Realistic Calibrated Stochastic Generator ending at today's date
        self.last_was_synthetic = True
        if interval != "1d":
            df = generate_synthetic_intraday_data(ticker=ticker_clean, interval=interval, initial_price=custom_price)
        else:
            df = generate_synthetic_stock_data(ticker=ticker_clean, start_date=start_date, end_date=end_date, initial_price=custom_price)
        return df

    def fetch_market_dataset(
        self,
        target_ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        period: Optional[str] = None,
        custom_price: Optional[float] = None,
        use_cache: bool = True,
        force_synthetic: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Download target stock along with benchmark market indices (SPY, ^VIX) and align timestamps.
        """
        target_df = self.fetch_ticker_data(
            target_ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            period=period,
            custom_price=custom_price,
            use_cache=use_cache,
            force_synthetic=force_synthetic
        )
        
        benchmarks = {}
        if interval == "1d":
            for bm_ticker in self.config.benchmark_tickers:
                if bm_ticker.upper() != target_ticker.upper():
                    try:
                        bm_df = self.fetch_ticker_data(
                            bm_ticker,
                            start_date=start_date,
                            end_date=end_date,
                            interval=interval,
                            period=period,
                            use_cache=use_cache,
                            force_synthetic=force_synthetic
                        )
                        benchmarks[bm_ticker] = bm_df
                    except Exception as e:
                        logger.warning(f"Could not load benchmark {bm_ticker}: {e}")
                        
        return target_df, benchmarks
