import os
import json
import time
import datetime
from typing import Dict, Any, List, Optional
from stock_predictor.execution.broker import BrokerAPI
from stock_predictor.execution.risk_manager import RiskManager

class PaperTrader:
    """
    Simulates real-world execution by receiving AI predictions
    and routing them through the Risk Manager and Broker.
    """
    
    def __init__(self, initial_capital: float = 100000.0, state_file: Optional[str] = None):
        self.broker = BrokerAPI(mode="paper")
        self.risk_manager = RiskManager(initial_capital=initial_capital)
        
        self.broker.connect()
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.state_file = state_file or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "portfolio.json"))
        
        self._load_state()
        print(f"Paper Trader Initialized. Capital: ${self.risk_manager.current_capital:,.2f}")

    def on_forecast_received(self, forecast: dict):
        """
        Process an incoming AI forecast and decide whether to trade.
        """
        ticker = str(forecast.get("ticker", "")).upper().strip()
        if not ticker:
            return
            
        signal = forecast.get("signal", "HOLD")
        confidence = float(forecast.get("confidence_score", 50.0))
        expected_return = float(forecast.get("predicted_return_pct", 0.0))
        current_price = float(forecast.get("current_price", 0.0))
        
        if current_price <= 0:
            return
            
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {ticker} Signal: {signal} (Conf: {confidence:.1f}%) Price: ${current_price:.2f}")
        
        if "BUY" in signal:
            self._handle_buy_signal(ticker, confidence, expected_return, current_price)
        elif "SELL" in signal:
            self._handle_sell_signal(ticker, current_price)

    def _handle_buy_signal(self, ticker: str, confidence: float, expected_return: float, current_price: float):
        # Already hold position?
        if ticker in self.positions:
            return
            
        # 1. Ask Risk Manager for position size
        allocate_dollar_amt = self.risk_manager.get_position_size(
            ticker=ticker,
            confidence_score=confidence,
            predicted_return_pct=expected_return
        )
        
        if allocate_dollar_amt <= 0:
            print(f" -> Risk Manager rejected {ticker} buy (allocation $0.00).")
            return
            
        # 2. Calculate Shares
        shares = int(allocate_dollar_amt / current_price)
        if shares <= 0:
            return
            
        # 3. Execute with live price
        try:
            order = self.broker.place_order(ticker, shares, "BUY", "MARKET", current_price=current_price)
            
            # Record simulated fill
            fill_price = float(order["filled_avg_price"])
            self.positions[ticker] = {
                "shares": shares,
                "entry_price": fill_price,
                "target_price": round(fill_price * (1.0 + expected_return / 100.0), 2),
                "stop_loss": round(fill_price * 0.97, 2)  # 3% dynamic stop loss
            }
            
            # Update Risk Manager capital (subtract cash)
            self.risk_manager.update_capital(self.risk_manager.current_capital - (shares * fill_price))
            
            print(f" -> Executed BUY {shares} {ticker} @ ${fill_price:.2f}")
            self._save_state()
            
        except Exception as e:
            print(f" -> Failed to execute buy: {e}")

    def _handle_sell_signal(self, ticker: str, current_price: Optional[float] = None):
        if ticker not in self.positions:
            return
            
        pos = self.positions[ticker]
        
        try:
            order = self.broker.place_order(ticker, pos["shares"], "SELL", "MARKET", current_price=current_price or pos.get("entry_price", 100.0))
            fill_price = float(order["filled_avg_price"])
            
            pnl = (fill_price - pos["entry_price"]) * pos["shares"]
            
            print(f" -> Executed SELL {pos['shares']} {ticker} @ ${fill_price:.2f}. PNL: ${pnl:+.2f}")
            
            # Add cash back to capital
            self.risk_manager.update_capital(self.risk_manager.current_capital + (pos["shares"] * fill_price))
            
            del self.positions[ticker]
            self.trade_history.append({
                "ticker": ticker,
                "pnl": round(pnl, 2),
                "exit_price": fill_price,
                "timestamp": datetime.datetime.now().isoformat()
            })
            self._save_state()
            
        except Exception as e:
            print(f" -> Failed to execute sell: {e}")

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.risk_manager.current_capital = float(state.get("capital", self.risk_manager.current_capital))
                    self.risk_manager.peak_capital = float(state.get("peak_capital", self.risk_manager.peak_capital))
                    self.risk_manager.initial_capital = float(state.get("initial_capital", self.risk_manager.initial_capital))
                    self.positions = state.get("positions", {})
                    self.trade_history = state.get("trade_history", [])
            except Exception as e:
                print(f"Notice: Could not load portfolio state: {e}")

    def _save_state(self):
        try:
            state = {
                "capital": round(self.risk_manager.current_capital, 2),
                "peak_capital": round(self.risk_manager.peak_capital, 2),
                "initial_capital": round(self.risk_manager.initial_capital, 2),
                "positions": self.positions,
                "trade_history": self.trade_history,
                "total_pnl": round(sum(t.get("pnl", 0.0) for t in self.trade_history), 2)
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Notice: Could not save portfolio state: {e}")
