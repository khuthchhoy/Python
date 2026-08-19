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


def test_api_watchlist(client):
    response = client.get("/api/watchlist?tickers=AAPL,NVDA")
    assert response.status_code == 200
    items = response.json()
    assert len(items) >= 1
    assert "ticker" in items[0]
    assert "predicted_price" in items[0]


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
