"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["self_learning"] == "Active"


def test_api_forecast_synthetic(client):
    response = client.get("/api/forecast?ticker=TSLA&synthetic=true&horizon=5&history_days=30")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TSLA"
    assert data["current_price"] > 0
    assert data["predicted_price"] > 0
    assert data["lower_bound_price"] <= data["upper_bound_price"]
    assert data["is_synthetic"] is True
    assert len(data["history"]) == 30
    assert data["execution_time_ms"] > 0
    
    # Check enriched analytics fields
    assert data["trade_plan"] is not None
    assert "action" in data["trade_plan"]
    assert "stop_loss" in data["trade_plan"]
    assert data["support_resistance"] is not None
    assert data["factor_scores"] is not None
    assert data["market_regime"] is not None
    assert data["analyst_report"] is not None
    assert data["learning_metrics"] is not None


def test_api_analyst_endpoint(client):
    response = client.get("/api/analyst?ticker=NVDA&timeframe=1w")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "NVDA"
    assert "verdict" in data
    assert "conviction_score" in data
    assert len(data["executive_summary"]) > 30
    assert len(data["primary_catalysts"]) > 0
    assert len(data["contrarian_risks"]) > 0


def test_api_learning_endpoint(client):
    response = client.get("/api/learning?ticker=NVDA")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "NVDA"
    assert "directional_accuracy_pct" in data
    assert "active_gbm_weight" in data
    assert "active_lstm_weight" in data


def test_api_screener(client):
    response = client.get("/api/screener?timeframe=1w")
    assert response.status_code == 200
    items = response.json()
    assert len(items) >= 5
    assert "ticker" in items[0]
    assert "current_price" in items[0]
    assert "change" in items[0]
    assert "change_pct" in items[0]
    assert "predicted_price" in items[0]
    assert "action" in items[0]
    assert "composite_score" in items[0]
    assert items[0]["current_price"] > 0
    assert items[0]["predicted_price"] > 0


def test_api_watchlist(client):
    response = client.get("/api/watchlist?tickers=AAPL,NVDA")
    assert response.status_code == 200
    items = response.json()
    assert len(items) >= 1
    assert "ticker" in items[0]
    assert "predicted_price" in items[0]
    assert "action" in items[0]


def test_api_quote(client):
    response = client.get("/api/quote?ticker=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["price"] > 0
    assert "change" in data
    assert "change_pct" in data


def test_api_backtest(client):
    response = client.get("/api/backtest?ticker=NVDA&timeframe=1w")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "equity_curve" in data
    assert data["summary"]["ticker"] == "NVDA"
