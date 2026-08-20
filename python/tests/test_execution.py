import os
import tempfile
import pytest
from stock_predictor.execution.risk_manager import RiskManager
from stock_predictor.execution.broker import BrokerAPI
from stock_predictor.execution.paper_trader import PaperTrader

def test_kelly_criterion():
    rm = RiskManager(initial_capital=100000.0, kelly_fraction=1.0)  # Full kelly for testing
    
    # Example: 60% win prob, 2:1 reward/risk
    # Kelly = 0.60 - (0.40 / 2.0) = 0.60 - 0.20 = 0.40 (40%)
    size_pct = rm.calculate_kelly_size(0.60, 2.0)
    assert abs(size_pct - 0.40) < 0.001
    
    # 50% win prob, 1:1 reward/risk -> Kelly = 0.50 - 0.50/1 = 0
    size_pct = rm.calculate_kelly_size(0.50, 1.0)
    assert size_pct == 0.0

def test_max_position_limit():
    rm = RiskManager(initial_capital=100000.0, max_position_pct=0.15)
    
    # High confidence trade that would give > 15% Kelly
    # 80% win prob, 3:1 reward/risk -> Kelly = 0.8 - (0.2/3) = 0.733.
    # Half kelly = 0.366
    alloc = rm.get_position_size("AAPL", 80.0, 6.0) # 6% return -> 3:1 reward/risk assuming 2% stop
    
    # Should be capped at 15% of 100,000 = 15,000
    assert alloc == 15000.0

def test_drawdown_halt():
    rm = RiskManager(initial_capital=100000.0, max_drawdown_pct=0.10)
    
    rm.update_capital(89000.0) # 11% drawdown
    assert rm.is_trading_halted() == True
    
    alloc = rm.get_position_size("AAPL", 99.0, 10.0)
    assert alloc == 0.0 # Halted

def test_broker_api_execution():
    broker = BrokerAPI(mode="paper")
    assert broker.connect() is True
    assert broker.get_account_balance() == 100000.0
    
    # Market order without limit price
    order = broker.place_order(ticker="NVDA", qty=10, side="BUY", order_type="MARKET", current_price=130.0)
    assert order["status"] == "filled"
    assert order["filled_qty"] == 10
    assert order["filled_avg_price"] == 130.0

    # Limit order
    order_lim = broker.place_order(ticker="NVDA", qty=5, side="SELL", order_type="LIMIT", limit_price=135.0)
    assert order_lim["status"] == "filled"
    assert order_lim["filled_avg_price"] == 135.0

def test_paper_trader_buy_sell_lifecycle():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        trader = PaperTrader(initial_capital=100000.0, state_file=tmp_path)
        
        # 1. Buy Signal
        forecast_buy = {
            "ticker": "NVDA",
            "signal": "STRONG_BUY",
            "confidence_score": 80.0,
            "predicted_return_pct": 5.0,
            "current_price": 130.0
        }
        trader.on_forecast_received(forecast_buy)
        
        assert "NVDA" in trader.positions
        nvda_pos = trader.positions["NVDA"]
        assert nvda_pos["shares"] > 0
        assert nvda_pos["entry_price"] == 130.0
        assert trader.risk_manager.current_capital < 100000.0
        
        # 2. Sell Signal
        forecast_sell = {
            "ticker": "NVDA",
            "signal": "STRONG_SELL",
            "confidence_score": 75.0,
            "predicted_return_pct": -4.0,
            "current_price": 140.0
        }
        trader.on_forecast_received(forecast_sell)
        
        assert "NVDA" not in trader.positions
        assert len(trader.trade_history) == 1
        assert trader.trade_history[0]["pnl"] > 0
        assert trader.risk_manager.current_capital > 100000.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
