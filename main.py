from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import yfinance as yf

from app.strategies.moving_average import MovingAverageCrossover
from app.strategies.rsi_strategy import RSIMeanReversion
from app.strategies.bollinger_bands import BollingerBandStrategy
from app.strategies.macd_strategy import MACDStrategy
from app.strategies.volume_momentum import VolumeMomentumStrategy
from app.backtester.engine import BacktestEngine
from app.backtester.metrics import PerformanceMetrics

app = FastAPI(title="Professional Backtesting Platform", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float = 10000
    parameters: Dict = {}
    timeframe: str = "1d"
    enable_risk_management: bool = True


class OptimizationRequest(BaseModel):
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float = 10000
    param_grid: Dict = {}
    metric: str = "sharpe_ratio"
    max_workers: int = 4


class BacktestResult(BaseModel):
    strategy_name: str
    symbol: str
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    var_5: float
    cvar_5: float
    total_trades: int
    equity_curve: List[Dict]
    trades: List[Dict]
    monthly_returns: List[Dict]
    price_data: List[Dict]  # Add price data for charts


@app.get("/")
async def root():
    return {"message": "Professional Backtesting Platform API"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Professional Backtesting Platform is running",
    }


@app.get("/api/strategies")
async def get_available_strategies():
    """Get list of available trading strategies"""
    return {
        "strategies": [
            {
                "id": "moving_average_crossover",
                "name": "Moving Average Crossover",
                "description": "Buy when fast MA crosses above slow MA, sell when crosses below",
                "parameters": {
                    "fast_period": {"type": "int", "default": 10, "min": 5, "max": 50},
                    "slow_period": {
                        "type": "int",
                        "default": 30,
                        "min": 20,
                        "max": 200,
                    },
                },
            },
            {
                "id": "rsi_mean_reversion",
                "name": "RSI Mean Reversion",
                "description": "Buy when RSI < oversold, sell when RSI > overbought",
                "parameters": {
                    "period": {"type": "int", "default": 14, "min": 5, "max": 30},
                    "oversold": {"type": "int", "default": 30, "min": 10, "max": 40},
                    "overbought": {"type": "int", "default": 70, "min": 60, "max": 90},
                },
            },
            {
                "id": "bollinger_bands",
                "name": "Bollinger Band Breakout",
                "description": "Buy on lower band touch, sell on upper band touch",
                "parameters": {
                    "period": {"type": "int", "default": 20, "min": 10, "max": 50},
                    "std_dev": {
                        "type": "float",
                        "default": 2.0,
                        "min": 1.0,
                        "max": 3.0,
                    },
                },
            },
            {
                "id": "macd_strategy",
                "name": "MACD Crossover",
                "description": "Buy when MACD crosses above signal line, sell when crosses below",
                "parameters": {
                    "fast_period": {"type": "int", "default": 12, "min": 5, "max": 20},
                    "slow_period": {"type": "int", "default": 26, "min": 20, "max": 50},
                    "signal_period": {"type": "int", "default": 9, "min": 5, "max": 15},
                },
            },
            {
                "id": "volume_momentum",
                "name": "Volume Momentum",
                "description": "Volume-confirmed momentum strategy",
                "parameters": {
                    "lookback": {"type": "int", "default": 20, "min": 10, "max": 50},
                    "volume_threshold": {
                        "type": "float",
                        "default": 1.5,
                        "min": 1.0,
                        "max": 3.0,
                    },
                    "momentum_threshold": {
                        "type": "float",
                        "default": 0.02,
                        "min": 0.01,
                        "max": 0.05,
                    },
                },
            },
        ]
    }


@app.get("/api/symbols")
async def get_available_symbols():
    """Get list of available symbols for backtesting"""
    return {
        "symbols": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
            {"symbol": "BTC-USD", "name": "Bitcoin USD"},
            {"symbol": "ETH-USD", "name": "Ethereum USD"},
        ]
    }


@app.get("/api/timeframes")
async def get_available_timeframes():
    """Get list of available timeframes for backtesting"""
    return {
        "timeframes": [
            {
                "id": "1h",
                "name": "1 Hour",
                "description": "1-hour candles",
                "category": "short_term",
                "recommended_period": "1-365 days",
            },
            {
                "id": "4h",
                "name": "4 Hours",
                "description": "4-hour candles",
                "category": "medium_term",
                "recommended_period": "1-365 days",
            },
            {
                "id": "1d",
                "name": "1 Day",
                "description": "Daily candles (default)",
                "category": "medium_term",
                "recommended_period": "1 month - 10 years",
            },
            {
                "id": "1wk",
                "name": "1 Week",
                "description": "Weekly candles",
                "category": "long_term",
                "recommended_period": "6 months - 20 years",
            },
            {
                "id": "1mo",
                "name": "1 Month",
                "description": "Monthly candles",
                "category": "long_term",
                "recommended_period": "2 years - 50 years",
            },
        ]
    }


@app.post("/api/backtest", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """Run a backtest for the specified strategy and parameters"""
    try:
        yf_interval = request.timeframe
        if not yf_interval:
            yf_interval = "1H"

        data = None
        try:
            ticker = yf.Ticker(request.symbol)
            data = ticker.history(
                start=request.start_date,
                end=request.end_date,
                interval=yf_interval,
                auto_adjust=True,
            )
            print(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{e}")

        if data.empty:
            print(f"No data available for symbol {request.symbol}")
            raise HTTPException(
                status_code=400, detail=f"No data available for symbol {request.symbol}"
            )

        # Standardize column names (yfinance sometimes returns different cases)
        data.columns = [col.lower() for col in data.columns]
        required_columns = ["open", "high", "low", "close", "volume"]

        for col in required_columns:
            if col not in data.columns:
                raise HTTPException(
                    status_code=400, detail=f"Missing required column: {col}"
                )

        print(f"Data sample:\n{data.head()}")

        # Initialize strategy
        strategy = None
        if request.strategy == "moving_average_crossover":
            strategy = MovingAverageCrossover(
                fast_period=request.parameters.get("fast_period", 10),
                slow_period=request.parameters.get("slow_period", 30),
            )
        elif request.strategy == "rsi_mean_reversion":
            strategy = RSIMeanReversion(
                period=request.parameters.get("period", 14),
                oversold=request.parameters.get("oversold", 30),
                overbought=request.parameters.get("overbought", 70),
            )
        elif request.strategy == "bollinger_bands":
            strategy = BollingerBandStrategy(
                period=request.parameters.get("period", 20),
                std_dev=request.parameters.get("std_dev", 2.0),
            )
        elif request.strategy == "macd_strategy":
            strategy = MACDStrategy(
                fast_period=request.parameters.get("fast_period", 12),
                slow_period=request.parameters.get("slow_period", 26),
                signal_period=request.parameters.get("signal_period", 9),
            )
        elif request.strategy == "volume_momentum":
            strategy = VolumeMomentumStrategy(
                lookback=request.parameters.get("lookback", 20),
                volume_threshold=request.parameters.get("volume_threshold", 1.5),
                momentum_threshold=request.parameters.get("momentum_threshold", 0.02),
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown strategy: {request.strategy}"
            )

        # Run backtest
        engine = BacktestEngine(initial_capital=request.initial_capital)
        results = engine.run_backtest(data, strategy)

        # Calculate performance metrics
        metrics = PerformanceMetrics(results)
        performance = metrics.calculate_all_metrics()

        # Format results for frontend
        equity_curve = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "value": float(value),
                "returns": float(returns),
            }
            for date, value, returns in zip(
                results["equity_curve"].index,
                results["equity_curve"]["portfolio_value"],
                results["equity_curve"]["returns"],
            )
        ]

        trades = [
            {
                "date": trade["date"].strftime("%Y-%m-%d"),
                "action": trade["action"],
                "price": float(trade["price"]),
                "quantity": int(trade["quantity"]),
                "value": float(trade["value"]),
                "pnl": float(trade.get("pnl", 0)),
            }
            for trade in results["trades"]
        ]

        # Calculate monthly returns
        monthly_returns = metrics.calculate_monthly_returns()
        monthly_returns_formatted = [
            {"month": month.strftime("%Y-%m"), "return": float(ret)}
            for month, ret in monthly_returns.items()
        ]

        # Format price data for charts
        price_data = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
            for date, row in data.iterrows()
        ]

        return BacktestResult(
            strategy_name=request.strategy,
            symbol=request.symbol,
            initial_capital=request.initial_capital,
            final_capital=float(performance["final_capital"]),
            total_return=float(performance["total_return"]),
            total_return_pct=float(performance["total_return_pct"]),
            sharpe_ratio=float(performance["sharpe_ratio"]),
            sortino_ratio=float(performance.get("sortino_ratio", 0)),
            calmar_ratio=float(performance.get("calmar_ratio", 0)),
            max_drawdown=float(performance["max_drawdown"]),
            win_rate=float(performance["win_rate"]),
            profit_factor=float(performance.get("profit_factor", 0)),
            var_5=float(performance.get("var_5", 0)),
            cvar_5=float(performance.get("cvar_5", 0)),
            total_trades=int(performance["total_trades"]),
            equity_curve=equity_curve,
            trades=trades,
            monthly_returns=monthly_returns_formatted,
            price_data=price_data,
        )

    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.get("/api/fundamentals")
def get_fundamentals(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        pros = []
        cons = []

        # Pros
        if info.get("trailingPE") and info["trailingPE"] < 20:
            pros.append("Low trailing P/E ratio (potentially undervalued)")
        if info.get("profitMargins") and info["profitMargins"] > 0.15:
            pros.append("High profit margins")
        if info.get("debtToEquity") and info["debtToEquity"] < 1:
            pros.append("Low debt-to-equity ratio (financially healthy)")
        if info.get("earningsGrowth") and info["earningsGrowth"] > 0.05:
            pros.append("Positive earnings growth")
        if info.get("revenueGrowth") and info["revenueGrowth"] > 0.05:
            pros.append("Positive revenue growth")
        if info.get("currentRatio") and info["currentRatio"] > 1.5:
            pros.append("Strong liquidity (high current ratio)")
        if info.get("dividendYield") and info["dividendYield"] > 0.02:
            pros.append("Attractive dividend yield")
        if info.get("beta") and info["beta"] < 1:
            pros.append("Low beta (less volatile than market)")
        # Cons
        if info.get("trailingPE") and info["trailingPE"] > 40:
            cons.append("High trailing P/E ratio (may indicate overvaluation)")
        if info.get("profitMargins") and info["profitMargins"] < 0.05:
            cons.append("Low profit margins")
        if info.get("debtToEquity") and info["debtToEquity"] > 2:
            cons.append("High debt-to-equity ratio (financial risk)")
        if info.get("earningsGrowth") and info["earningsGrowth"] < 0:
            cons.append("Negative earnings growth")
        if info.get("revenueGrowth") and info["revenueGrowth"] < 0:
            cons.append("Negative revenue growth")
        if info.get("currentRatio") and info["currentRatio"] < 1:
            cons.append("Low current ratio (liquidity risk)")
        if info.get("beta") and info["beta"] > 2:
            cons.append("High beta (stock is very volatile)")
        return {
            "symbol": symbol,
            "longName": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "dividendYield": info.get("dividendYield"),
            "earningsGrowth": info.get("earningsGrowth"),
            "revenueGrowth": info.get("revenueGrowth"),
            "profitMargins": info.get("profitMargins"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnAssets": info.get("returnOnAssets"),
            "debtToEquity": info.get("debtToEquity"),
            "currentRatio": info.get("currentRatio"),
            "quickRatio": info.get("quickRatio"),
            "beta": info.get("beta"),
            "website": info.get("website"),
            "pros": pros,
            "cons": cons,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/news")
def get_stock_news(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news if hasattr(ticker, "news") else []
        # Get top 5 latest news
        top_news = sorted(
            news, key=lambda n: n.get("providerPublishTime", 0), reverse=True
        )[:5]
        flat_news = []
        for n in top_news:
            # If 'content' key exists, flatten it
            content = n.get("content", {})
            flat_news.append(
                {
                    "title": content.get("title", n.get("title")),
                    "link": content.get("canonicalUrl", {}).get("url")
                    or content.get("clickThroughUrl", {}).get("url")
                    or n.get("link"),
                    "publisher": content.get("provider", {}).get(
                        "displayName", n.get("publisher")
                    ),
                    "providerPublishTime": content.get("pubDate")
                    or n.get("providerPublishTime"),
                    "summary": content.get("summary", n.get("summary", "")),
                    "thumbnail": content.get("thumbnail", {}).get("originalUrl", None),
                }
            )
        return flat_news
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
