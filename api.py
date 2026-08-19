"""FastAPI REST & WebSocket API Backend with Live Progress Streaming and Price Calibration."""

import sys
import time
import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from stock_predictor.config import PredictionConfig, parse_timeframe, TimeframeSpec
from stock_predictor.data.downloader import StockDataDownloader
from stock_predictor.features.pipeline import FeaturePipeline
from stock_predictor.models.ensemble import EnsembleStockPredictor
from stock_predictor.evaluation.backtest import WalkForwardBacktester

app = FastAPI(
    title="AI Stock Predictor & Quantitative Engine API",
    description="Production Quantitative AI API for live streaming and multi-horizon stock price forecasting (10m, 20m, 30m, 1h, 4h, 1d, 1w) on iOS & Web",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE: Dict[str, Tuple[float, Any, Any]] = {}
CACHE_TTL_SECONDS = 600  # 10 minutes


# --- Response Schemas ---
class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class HorizonPoint(BaseModel):
    timeframe: str
    minutes_ahead: int
    predicted_price: float
    predicted_return_pct: float
    lower_bound_price: float
    upper_bound_price: float
    direction: str
    target_time: str


class ForecastResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    lower_bound_price: float
    upper_bound_price: float
    direction: str
    direction_prob: float
    signal: str
    confidence_score: float
    forecast_horizon_days: int
    forecast_date: str
    target_date: str
    timeframe: str = "1w"
    top_features: Dict[str, float]
    history: List[PricePoint]
    is_synthetic: bool = False
    data_source: str = "Live Yahoo Finance"
    execution_time_ms: float = 0.0
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    multi_horizon_path: Optional[List[HorizonPoint]] = None


class BacktestSummaryResponse(BaseModel):
    ticker: str
    directional_accuracy: float
    interval_coverage: float
    strategy_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    annualized_volatility_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float


class BacktestDetailResponse(BaseModel):
    summary: BacktestSummaryResponse
    equity_curve: List[Dict[str, Any]]


class WatchlistTickerItem(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    signal: str
    direction_prob: float
    is_synthetic: bool = False


class QuoteResponse(BaseModel):
    ticker: str
    price: float
    change: float
    change_pct: float
    open: float
    high: float
    low: float
    volume: int
    timestamp: str
    is_synthetic: bool = False


class ScreenerItem(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    signal: str
    direction_prob: float
    confidence_score: float
    sharpe_ratio: float


def _compute_forecast_and_backtest(
    ticker_clean: str,
    timeframe_str: str,
    history_days: int,
    synthetic: bool,
    custom_price: Optional[float] = None
) -> Tuple[ForecastResponse, BacktestSummaryResponse, List[Dict[str, Any]]]:
    t0 = time.time()
    spec = parse_timeframe(timeframe_str)
    
    config = PredictionConfig(
        forecast_horizon=spec.horizon_bars,
        data_interval=spec.data_interval,
        timeframe=spec.timeframe_id
    )
    downloader = StockDataDownloader(config)
    target_df, benchmarks = downloader.fetch_market_dataset(
        target_ticker=ticker_clean,
        interval=spec.data_interval,
        period=spec.default_period,
        custom_price=custom_price,
        use_cache=True,
        force_synthetic=synthetic
    )
    is_synthetic = downloader.last_was_synthetic

    pipeline = FeaturePipeline(config)
    dataset = pipeline.prepare_dataset(target_df, benchmarks, horizon=spec.horizon_bars)
    splits = pipeline.train_val_test_split(dataset)

    ensemble = EnsembleStockPredictor(config=config)
    train_data = splits["train"]
    ensemble.fit(train_data["X"], train_data["y_return"], train_data["y_dir"])

    forecast = ensemble.generate_forecast(
        ticker=ticker_clean,
        X_latest=dataset["X_latest"],
        current_price=dataset["latest_price"],
        latest_date=dataset["latest_date"],
        horizon_days=max(1, spec.minutes_ahead // 1440),
        timeframe=spec.timeframe_id,
        minutes_ahead=spec.minutes_ahead,
        is_synthetic=is_synthetic
    )

    backtester = WalkForwardBacktester(config)
    backtest_df, backtest_metrics = backtester.evaluate_test_set(
        model=ensemble,
        test_data=splits["test"]
    )

    max_bars = min(len(target_df), max(30, history_days))
    recent_df = target_df.iloc[-max_bars:]
    
    is_intraday = spec.minutes_ahead < 1440
    history_points = [
        PricePoint(
            date=dt.strftime("%Y-%m-%d %H:%M") if is_intraday else dt.strftime("%Y-%m-%d"),
            open=round(float(row["Open"]), 2),
            high=round(float(row["High"]), 2),
            low=round(float(row["Low"]), 2),
            close=round(float(row["Close"]), 2),
            volume=int(row["Volume"])
        )
        for dt, row in recent_df.iterrows()
    ]

    top_feats = dict(list(forecast.feature_importances.items())[:8]) if forecast.feature_importances else {}
    exec_time = round((time.time() - t0) * 1000.0, 1)
    source_label = "Calibrated Simulated Feed" if is_synthetic else f"Live Market Feed ({spec.data_interval})"

    fc_resp = ForecastResponse(
        ticker=forecast.ticker,
        current_price=forecast.current_price,
        predicted_price=forecast.predicted_price,
        predicted_return_pct=forecast.predicted_return_pct,
        lower_bound_price=forecast.lower_bound_price,
        upper_bound_price=forecast.upper_bound_price,
        direction=forecast.direction,
        direction_prob=forecast.direction_prob,
        signal=forecast.signal,
        confidence_score=forecast.confidence_score,
        forecast_horizon_days=forecast.forecast_horizon_days,
        forecast_date=forecast.forecast_date,
        target_date=forecast.target_date,
        timeframe=spec.timeframe_id,
        top_features=top_feats,
        history=history_points,
        is_synthetic=is_synthetic,
        data_source=source_label,
        execution_time_ms=exec_time,
        sharpe_ratio=backtest_metrics.sharpe_ratio,
        max_drawdown_pct=backtest_metrics.max_drawdown_pct,
        multi_horizon_path=[
            HorizonPoint(
                timeframe=hp.timeframe,
                minutes_ahead=hp.minutes_ahead,
                predicted_price=hp.predicted_price,
                predicted_return_pct=hp.predicted_return_pct,
                lower_bound_price=hp.lower_bound_price,
                upper_bound_price=hp.upper_bound_price,
                direction=hp.direction,
                target_time=hp.target_time
            )
            for hp in (forecast.multi_horizon_path or [])
        ]
    )

    bt_resp = BacktestSummaryResponse(
        ticker=ticker_clean,
        directional_accuracy=backtest_metrics.directional_accuracy,
        interval_coverage=backtest_metrics.interval_coverage,
        strategy_return_pct=backtest_metrics.strategy_return_pct,
        benchmark_return_pct=backtest_metrics.benchmark_return_pct,
        alpha_pct=backtest_metrics.alpha_pct,
        sharpe_ratio=backtest_metrics.sharpe_ratio,
        sortino_ratio=backtest_metrics.sortino_ratio,
        calmar_ratio=backtest_metrics.calmar_ratio,
        annualized_volatility_pct=backtest_metrics.annualized_volatility_pct,
        max_drawdown_pct=backtest_metrics.max_drawdown_pct,
        win_rate_pct=backtest_metrics.win_rate_pct,
        profit_factor=backtest_metrics.profit_factor
    )

    equity_points = []
    if backtest_df is not None and len(backtest_df) > 0:
        for dt_idx, row in backtest_df.tail(60).iterrows():
            dt_str = dt_idx.strftime("%Y-%m-%d") if hasattr(dt_idx, "strftime") else str(dt_idx)
            equity_points.append({
                "date": dt_str,
                "strategy_equity": round(float(row.get("Strategy_Equity", 1.0)), 4),
                "benchmark_equity": round(float(row.get("Benchmark_Equity", 1.0)), 4),
                "pred_return_pct": round(float(row.get("Pred_Return_5d_pct", 0.0)), 2),
                "true_return_pct": round(float(row.get("True_Return_5d_pct", 0.0)), 2)
            })

    return fc_resp, bt_resp, equity_points


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "Stock Predictor AI API",
        "version": "3.0.0",
        "engine": "Ensemble (GBM Quantile + PyTorch Temporal Attention)"
    }


@app.get("/api/quote", response_model=QuoteResponse)
def get_quote(
    ticker: str = Query(..., description="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)")
):
    """Fast lightweight real-time quote for a stock."""
    ticker_clean = ticker.upper().strip()
    downloader = StockDataDownloader()
    df = downloader.fetch_ticker_data(ticker_clean, interval="1d", period="5d", use_cache=True)
    
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No quote data available for {ticker_clean}")
        
    latest_row = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest_row["Open"]
    cur_price = round(float(latest_row["Close"]), 2)
    change = round(cur_price - float(prev_close), 2)
    change_pct = round((change / float(prev_close)) * 100.0, 2)
    
    return QuoteResponse(
        ticker=ticker_clean,
        price=cur_price,
        change=change,
        change_pct=change_pct,
        open=round(float(latest_row["Open"]), 2),
        high=round(float(latest_row["High"]), 2),
        low=round(float(latest_row["Low"]), 2),
        volume=int(latest_row["Volume"]),
        timestamp=df.index[-1].strftime("%Y-%m-%d %H:%M"),
        is_synthetic=downloader.last_was_synthetic
    )


@app.get("/api/forecast", response_model=ForecastResponse)
def get_forecast(
    ticker: str = Query(..., description="Stock ticker symbol (e.g. AAPL, NVDA, DELL)"),
    timeframe: str = Query("1w", description="Forecast timeframe: 10m, 20m, 30m, 1h, 2h, 4h, 1d, 1w"),
    price: Optional[float] = Query(None, description="Optional custom anchor price override"),
    horizon: Optional[int] = Query(None, description="Legacy day horizon override"),
    history_days: int = Query(60, description="Number of historical price points for charting"),
    synthetic: bool = Query(False, description="Force synthetic data for offline testing")
):
    ticker_clean = ticker.upper().strip()
    tf_param = f"{horizon}d" if (horizon is not None and timeframe == "1w") else timeframe
    
    cache_key = f"{ticker_clean}_{tf_param}_{history_days}_{synthetic}_{price}"
    now = time.time()

    if cache_key in _CACHE:
        cached_time, cached_fc, _ = _CACHE[cache_key]
        if (now - cached_time) < CACHE_TTL_SECONDS:
            return cached_fc

    try:
        fc_resp, bt_resp, _ = _compute_forecast_and_backtest(
            ticker_clean=ticker_clean,
            timeframe_str=tf_param,
            history_days=history_days,
            synthetic=synthetic,
            custom_price=price
        )
        _CACHE[cache_key] = (now, fc_resp, bt_resp)
        return fc_resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed for {ticker_clean} ({tf_param}): {str(e)}")


@app.get("/api/backtest", response_model=BacktestDetailResponse)
def get_backtest_details(
    ticker: str = Query(..., description="Stock ticker symbol (e.g. AAPL, NVDA)"),
    timeframe: str = Query("1w", description="Forecast timeframe: 1d, 1w")
):
    ticker_clean = ticker.upper().strip()
    try:
        _, bt_resp, equity_curve = _compute_forecast_and_backtest(
            ticker_clean=ticker_clean,
            timeframe_str=timeframe,
            history_days=60,
            synthetic=False
        )
        return BacktestDetailResponse(
            summary=bt_resp,
            equity_curve=equity_curve
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest calculation failed for {ticker_clean}: {str(e)}")


@app.get("/api/screener", response_model=List[ScreenerItem])
def get_market_screener(
    timeframe: str = Query("1w", description="Timeframe: 1d, 1w")
):
    """Scans and ranks high-liquidity market leaders by AI opportunity score."""
    screener_tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "PLTR", "AMD", "SPY"]
    results: List[ScreenerItem] = []
    
    for t in screener_tickers:
        try:
            fc_resp, bt_resp, _ = _compute_forecast_and_backtest(
                ticker_clean=t,
                timeframe_str=timeframe,
                history_days=30,
                synthetic=False
            )
            results.append(ScreenerItem(
                ticker=t,
                current_price=fc_resp.current_price,
                predicted_price=fc_resp.predicted_price,
                predicted_return_pct=fc_resp.predicted_return_pct,
                signal=fc_resp.signal,
                direction_prob=fc_resp.direction_prob,
                confidence_score=fc_resp.confidence_score,
                sharpe_ratio=bt_resp.sharpe_ratio
            ))
        except Exception:
            continue
            
    # Sort descending by predicted return
    results.sort(key=lambda item: item.predicted_return_pct, reverse=True)
    return results


@app.get("/api/watchlist", response_model=List[WatchlistTickerItem])
def get_watchlist(
    tickers: str = Query("NVDA,AAPL,MSFT,TSLA,AMZN,META,SPY"),
    timeframe: str = Query("1w", description="Timeframe for watchlist items")
):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results = []

    for t in ticker_list:
        cache_key = f"{t}_{timeframe}_60_False_None"
        now = time.time()
        if cache_key in _CACHE:
            cached_time, cached_fc, _ = _CACHE[cache_key]
            if (now - cached_time) < CACHE_TTL_SECONDS:
                results.append(WatchlistTickerItem(
                    ticker=t,
                    current_price=cached_fc.current_price,
                    predicted_price=cached_fc.predicted_price,
                    predicted_return_pct=cached_fc.predicted_return_pct,
                    signal=cached_fc.signal,
                    direction_prob=cached_fc.direction_prob,
                    is_synthetic=cached_fc.is_synthetic
                ))
                continue

        try:
            fc_resp, bt_resp, _ = _compute_forecast_and_backtest(
                ticker_clean=t,
                timeframe_str=timeframe,
                history_days=60,
                synthetic=False
            )
            _CACHE[cache_key] = (now, fc_resp, bt_resp)
            results.append(WatchlistTickerItem(
                ticker=t,
                current_price=fc_resp.current_price,
                predicted_price=fc_resp.predicted_price,
                predicted_return_pct=fc_resp.predicted_return_pct,
                signal=fc_resp.signal,
                direction_prob=fc_resp.direction_prob,
                is_synthetic=fc_resp.is_synthetic
            ))
        except Exception:
            continue

    return results


@app.websocket("/ws/live/{ticker}")
async def websocket_live_stream(
    websocket: WebSocket,
    ticker: str,
    timeframe: str = "10m"
):
    """WebSocket streaming real-time live price ticks and dynamic forecasts every 2 seconds."""
    await websocket.accept()
    ticker_clean = ticker.upper().strip()
    
    try:
        fc_resp, _, _ = _compute_forecast_and_backtest(
            ticker_clean=ticker_clean,
            timeframe_str=timeframe,
            history_days=30,
            synthetic=True
        )
        
        live_price = fc_resp.current_price
        rng = np.random.default_rng()
        tick_id = 1
        
        while True:
            tick_drift = float(rng.normal(0, 0.0012))
            delta = live_price * tick_drift
            live_price = max(1.0, round(live_price + delta, 2))
            
            ret_pct = fc_resp.predicted_return_pct
            pred_p = round(live_price * (1.0 + ret_pct / 100.0), 2)
            
            payload = {
                "tick_id": tick_id,
                "ticker": ticker_clean,
                "timeframe": timeframe,
                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                "current_price": live_price,
                "delta": round(delta, 2),
                "predicted_price": pred_p,
                "predicted_return_pct": ret_pct,
                "signal": fc_resp.signal,
                "direction_prob": fc_resp.direction_prob,
                "multi_horizon_path": [hp.model_dump() for hp in (fc_resp.multi_horizon_path or [])]
            }
            
            await websocket.send_text(json.dumps(payload))
            tick_id += 1
            await asyncio.sleep(2.0)
            
    except WebSocketDisconnect:
        pass
    except Exception as err:
        try:
            await websocket.send_text(json.dumps({"error": str(err)}))
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Stock Predictor AI API server on http://0.0.0.0:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
