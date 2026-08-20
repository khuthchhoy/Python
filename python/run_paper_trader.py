import time
import datetime
from api import get_forecast
from stock_predictor.execution.paper_trader import PaperTrader

def main():
    print("========================================")
    print("🤖 STARTING AUTONOMOUS PAPER TRADER 🤖")
    print("========================================")
    
    # Initialize the simulated trading engine with $100,000
    trader = PaperTrader(initial_capital=100000.0)
    
    # The portfolio of stocks we want the AI to autonomously trade
    tickers_to_trade = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "META", "SPY"]
    
    print(f"Tracking {len(tickers_to_trade)} assets. Waiting for market signals...")
    
    try:
        while True:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n--- Market Scan @ {current_time} ---")
            
            for ticker in tickers_to_trade:
                try:
                    # 1. Ask the AI model for its prediction
                    # Using synthetic=False to get real live market quotes
                    forecast_obj = get_forecast(
                        ticker=ticker,
                        timeframe="10m",  # Looking for short-term trades
                        history_days=60,
                        synthetic=False
                    )
                    
                    # 2. Convert the complex forecast object into a dictionary 
                    # so the PaperTrader can read it easily.
                    forecast_dict = {
                        "ticker": forecast_obj.ticker,
                        "signal": forecast_obj.signal,
                        "confidence_score": forecast_obj.confidence_score,
                        "predicted_return_pct": forecast_obj.predicted_return_pct,
                        "current_price": forecast_obj.current_price
                    }
                    
                    # 3. Feed the signal into our execution engine!
                    trader.on_forecast_received(forecast_dict)
                    
                except Exception as e:
                    print(f"[!] Error processing {ticker}: {e}")
                    
                # Small delay between stocks to prevent API rate limits
                time.sleep(2)
                
            print("\n[Zzz] Scan complete. Sleeping for 1 minute before next scan...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n🛑 Trading Halted by User.")
        
        # Print a final P&L summary
        print("\n=== FINAL PAPER TRADING P&L ===")
        total_pnl = sum(trade["pnl"] for trade in trader.trade_history)
        print(f"Total Closed P&L: ${total_pnl:.2f}")
        print(f"Remaining Open Positions: {list(trader.positions.keys())}")

if __name__ == "__main__":
    main()
