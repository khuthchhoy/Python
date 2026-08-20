"""Main entry point for the 1-Week Ahead Stock Price Prediction System."""

import argparse
import sys
import subprocess
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from cli import run_prediction


def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction AI Engine")
    parser.add_argument("--web", action="store_true", help="Launch the interactive Streamlit Web UI")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g. NVDA, AAPL, MSFT, TSLA)")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days (default: 5)")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Historical data start date (YYYY-MM-DD)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic market data")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit Web UI")

    args = parser.parse_args()

    if args.web:
        dashboard_path = root_dir / "stock_predictor" / "app" / "dashboard.py"
        print(f"🚀 Launching Streamlit Web UI on port {args.port}...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(args.port),
            "--server.headless", "true"
        ])
    else:
        tf = f"{args.horizon}d" if args.horizon else "1w"
        run_prediction(
            ticker=args.ticker,
            timeframe=tf,
            start_date=args.start_date,
            force_synthetic=args.synthetic
        )


if __name__ == "__main__":
    main()
