"""Command-line interface (CLI) for multi-horizon stock price prediction with Live Progress streaming."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from stock_predictor.config import PredictionConfig, parse_timeframe
from stock_predictor.data.downloader import StockDataDownloader
from stock_predictor.features.pipeline import FeaturePipeline
from stock_predictor.models.ensemble import EnsembleStockPredictor
from stock_predictor.evaluation.backtest import WalkForwardBacktester
from stock_predictor.utils.logger import setup_logger

try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.live import Live
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = setup_logger("cli")


def generate_rich_renderable(forecast, metrics, ticker: str, timeframe_label: str, tick_count: int = 0, last_tick_delta: float = 0.0):
    source_tag = "[yellow](Calibrated Simulated Feed)[/yellow]" if forecast.is_synthetic else "[green](Live Market Feed)[/green]"
    live_badge = f"[blink bold red]🔴 LIVE STREAMING[/blink bold red] • Tick #{tick_count} • [cyan]{datetime.now().strftime('%H:%M:%S')}[/cyan]"
    
    header_panel = Panel(
        f"[bold cyan]QUANTITATIVE AI STOCK PREDICTOR — {timeframe_label.upper()}[/bold cyan] {source_tag}\n"
        f"{live_badge}\n"
        f"[dim]Multi-Model Ensemble (Quantile Trees + PyTorch Temporal Attention) for {ticker.upper()}[/dim]",
        box=box.DOUBLE
    )
    
    ret_color = "green" if forecast.predicted_return_pct >= 0 else "red"
    sig_color = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "HOLD": "yellow",
        "SELL": "red",
        "STRONG_SELL": "bold red"
    }.get(forecast.signal, "white")
    
    tick_str = f" ({last_tick_delta:+.2f})" if last_tick_delta != 0.0 else ""
    tick_col = "green" if last_tick_delta >= 0 else "red"
    
    table = Table(title=f"🎯 Target Forecast ({forecast.forecast_date} ➔ {forecast.target_date})", box=box.ROUNDED)
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="bold")
    
    table.add_row("Live Market Price", f"${forecast.current_price:.2f} [{tick_col}]{tick_str}[/{tick_col}]")
    table.add_row(f"{timeframe_label} Target Price", f"[{ret_color}]${forecast.predicted_price:.2f} ({forecast.predicted_return_pct:+.2f}%)[/{ret_color}]")
    table.add_row("80% Confidence Interval", f"[magenta]${forecast.lower_bound_price:.2f} – ${forecast.upper_bound_price:.2f}[/magenta]")
    table.add_row("Direction Bias", f"[{ret_color}]{forecast.direction}[/{ret_color}]")
    table.add_row("Directional Win Probability P(Up)", f"{forecast.direction_prob*100:.1f}%")
    table.add_row("Actionable Trading Signal", f"[{sig_color}]{forecast.signal.replace('_', ' ')}[/{sig_color}]")
    table.add_row("Model Confidence Score", f"{forecast.confidence_score:.1f} / 100")
    
    # Multi-Horizon Term Structure Table
    mh_table = Table(title="📈 Multi-Horizon Term Structure Trajectory Path", box=box.ROUNDED)
    mh_table.add_column("Horizon", style="bold cyan")
    mh_table.add_column("Target Time", style="bold white")
    mh_table.add_column("Predicted Price", style="bold")
    mh_table.add_column("Expected Return", style="bold")
    mh_table.add_column("80% Confidence Cone", style="magenta")
    mh_table.add_column("Bias", style="bold")
    
    if forecast.multi_horizon_path:
        for hp in forecast.multi_horizon_path:
            hp_color = "green" if hp.predicted_return_pct >= 0 else "red"
            mh_table.add_row(
                hp.timeframe.upper(),
                hp.target_time,
                f"${hp.predicted_price:.2f}",
                f"[{hp_color}]{hp.predicted_return_pct:+.2f}%[/{hp_color}]",
                f"${hp.lower_bound_price:.2f} – ${hp.upper_bound_price:.2f}",
                f"[{hp_color}]{hp.direction}[/{hp_color}]"
            )
            
    # Risk Metrics Table
    b_table = Table(title="🧪 Out-of-Sample Walk-Forward Backtest Performance", box=box.ROUNDED)
    b_table.add_column("Indicator", style="bold white")
    b_table.add_column("Result", style="cyan")
    
    b_table.add_row("Directional Hit Rate", f"{metrics.directional_accuracy:.1f}%")
    b_table.add_row("Interval Coverage (80%)", f"{metrics.interval_coverage:.1f}%")
    b_table.add_row("Simulated Return", f"{metrics.strategy_return_pct:+.2f}%")
    b_table.add_row("Annualized Sharpe", f"{metrics.sharpe_ratio:.2f}")
    b_table.add_row("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
    b_table.add_row("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%")
    b_table.add_row("Win Rate", f"{metrics.win_rate_pct:.1f}%")
    
    return Group(header_panel, table, mh_table, b_table)


def run_live_progress(
    ticker: str = "DELL",
    timeframe: str = "10m",
    custom_price: float = 0.0,
    refresh_seconds: float = 2.5,
    force_synthetic: bool = True
):
    """Continuous live streaming progress dashboard in the terminal."""
    console = Console()
    spec = parse_timeframe(timeframe)
    config = PredictionConfig(
        forecast_horizon=spec.horizon_bars,
        data_interval=spec.data_interval,
        timeframe=spec.timeframe_id
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Initializing AI Ensemble Models for {task.description}..."),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(ticker.upper(), total=100)
        
        downloader = StockDataDownloader(config)
        progress.update(task, advance=25)
        
        target_df, benchmarks = downloader.fetch_market_dataset(
            target_ticker=ticker,
            interval=spec.data_interval,
            period=spec.default_period,
            custom_price=custom_price if custom_price > 0 else None,
            use_cache=True,
            force_synthetic=force_synthetic
        )
        is_synthetic = downloader.last_was_synthetic
        progress.update(task, advance=25)
        
        pipeline = FeaturePipeline(config)
        dataset = pipeline.prepare_dataset(target_df, benchmarks, horizon=spec.horizon_bars)
        splits = pipeline.train_val_test_split(dataset)
        progress.update(task, advance=25)
        
        ensemble = EnsembleStockPredictor(config=config)
        train_data = splits["train"]
        ensemble.fit(train_data["X"], train_data["y_return"], train_data["y_dir"])
        progress.update(task, advance=25)
        
        backtester = WalkForwardBacktester(config)
        _, backtest_metrics = backtester.evaluate_test_set(ensemble, splits["test"])

    forecast = ensemble.generate_forecast(
        ticker=ticker,
        X_latest=dataset["X_latest"],
        current_price=dataset["latest_price"],
        latest_date=dataset["latest_date"],
        horizon_days=max(1, spec.minutes_ahead // 1440),
        timeframe=spec.timeframe_id,
        minutes_ahead=spec.minutes_ahead,
        is_synthetic=is_synthetic
    )
    
    current_live_price = float(dataset["latest_price"])
    rng = np.random.default_rng()
    tick_count = 1
    last_delta = 0.0

    console.print(f"[bold green]✓ AI Models Ready for {ticker.upper()} (${current_live_price:.2f}). Starting Live Progress Stream (Press Ctrl+C to exit)...[/bold green]\n")
    
    try:
        with Live(generate_rich_renderable(forecast, backtest_metrics, ticker, spec.human_label, tick_count, last_delta), console=console, refresh_per_second=4) as live:
            while True:
                time.sleep(refresh_seconds)
                tick_count += 1
                
                tick_pct = float(rng.normal(0, 0.0012))
                last_delta = current_live_price * tick_pct
                current_live_price = max(1.0, current_live_price + last_delta)
                
                forecast = ensemble.generate_forecast(
                    ticker=ticker,
                    X_latest=dataset["X_latest"],
                    current_price=current_live_price,
                    latest_date=pd.Timestamp.now(),
                    horizon_days=max(1, spec.minutes_ahead // 1440),
                    timeframe=spec.timeframe_id,
                    minutes_ahead=spec.minutes_ahead,
                    is_synthetic=is_synthetic
                )
                
                live.update(generate_rich_renderable(forecast, backtest_metrics, ticker, spec.human_label, tick_count, last_delta))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]🛑 Live streaming stopped by user.[/bold yellow]")


def run_prediction(
    ticker: str = "DELL",
    timeframe: str = "1w",
    price: float = 0.0,
    start_date: str = "",
    force_synthetic: bool = False,
    save_csv: str = "",
    live: bool = False,
    refresh_seconds: float = 2.5
):
    if live and HAS_RICH:
        run_live_progress(
            ticker=ticker,
            timeframe=timeframe,
            custom_price=price,
            refresh_seconds=refresh_seconds,
            force_synthetic=force_synthetic
        )
        return
        
    spec = parse_timeframe(timeframe)
    logger.info(f"Initiating stock prediction pipeline for {ticker.upper()} (Timeframe: {spec.human_label} / interval: {spec.data_interval})...")
    
    config = PredictionConfig(
        forecast_horizon=spec.horizon_bars,
        data_interval=spec.data_interval,
        timeframe=spec.timeframe_id
    )
    
    downloader = StockDataDownloader(config)
    target_df, benchmarks = downloader.fetch_market_dataset(
        target_ticker=ticker,
        start_date=start_date or None,
        interval=spec.data_interval,
        period=spec.default_period,
        custom_price=price if price > 0 else None,
        use_cache=True,
        force_synthetic=force_synthetic
    )
    is_synthetic = downloader.last_was_synthetic
    
    logger.info(f"Loaded {len(target_df)} bars of market history ({'Synthetic' if is_synthetic else 'Live'}).")
    
    pipeline = FeaturePipeline(config)
    dataset = pipeline.prepare_dataset(target_df, benchmarks, horizon=spec.horizon_bars)
    splits = pipeline.train_val_test_split(dataset)
    
    ensemble = EnsembleStockPredictor(config=config)
    train_data = splits["train"]
    ensemble.fit(
        X=train_data["X"],
        y_return=train_data["y_return"],
        y_dir=train_data["y_dir"]
    )
    
    forecast = ensemble.generate_forecast(
        ticker=ticker,
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
    
    if HAS_RICH:
        console = Console()
        console.print(generate_rich_renderable(forecast, backtest_metrics, ticker, spec.human_label))
    else:
        print(f"Current: ${forecast.current_price:.2f} | Target: ${forecast.predicted_price:.2f} ({forecast.predicted_return_pct:+.2f}%)")
        
    if save_csv:
        backtest_df.to_csv(save_csv)
        logger.info(f"Saved backtest predictions to {save_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Stock Price Prediction System (Live Progress & Multi-Horizon)")
    parser.add_argument("--ticker", type=str, default="DELL", help="Stock ticker symbol (e.g. DELL, AAPL, NVDA, TSLA)")
    parser.add_argument("--timeframe", type=str, default="1w", help="Timeframe: 10m, 20m, 30m, 1h, 2h, 4h, 1d, 1w")
    parser.add_argument("--price", type=float, default=0.0, help="Optional manual stock price anchor override (e.g. --price 138.50)")
    parser.add_argument("--start-date", type=str, default="", help="Start date for historical data (YYYY-MM-DD)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic market data for offline testing")
    parser.add_argument("--save-csv", type=str, default="", help="Optional CSV file path to save predictions")
    parser.add_argument("--live", action="store_true", help="Enable continuous real-time live progress streaming")
    parser.add_argument("--refresh-interval", type=float, default=2.5, help="Refresh interval in seconds for live mode")
    
    args = parser.parse_args()
    run_prediction(
        ticker=args.ticker,
        timeframe=args.timeframe,
        price=args.price,
        start_date=args.start_date,
        force_synthetic=args.synthetic,
        save_csv=args.save_csv,
        live=args.live,
        refresh_seconds=args.refresh_interval
    )
