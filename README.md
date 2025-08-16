# Backtesting Platform Backend

This is the backend for the Backtesting Platform, designed to support financial strategy backtesting, fundamental analysis, and news integration. It is implemented in Python and structured for modularity and extensibility.

## Features
- **Backtesting Engine**: Run and evaluate trading strategies using historical data.
- **Strategy Modules**: Includes Bollinger Bands, MACD, Moving Average, RSI, Volume Momentum, and a base strategy for custom implementations.
- **Fundamental Analysis**: Analyze financial fundamentals of assets.
- **News Integration**: Fetch and process financial news for sentiment and event analysis.
- **Metrics Calculation**: Compute performance metrics for strategies.

## Tech Stack & Dependencies

- **Python 3.11+**: Main programming language.
- **Standard Libraries**: Used for core logic, data handling, and file operations.
- **Third-party Libraries**: (see `requirements.txt` for full list)
   - Commonly used: `pandas`, `numpy`, `requests`, etc.
- **Cerebrium**: For deployment configuration (`cerebrium.toml`).
- **Vercel**: For cloud deployment integration (`.vercel/project.json`).

## Project Structure
```
backtesting-platform-backend/
├── main.py                  # Entry point for backend service
├── requirements.txt         # Python dependencies
├── cerebrium.toml           # Cerebrium deployment config
├── app/
│   ├── fundamentals.py      # Fundamental analysis logic
│   ├── news.py              # News processing logic
│   ├── backtester/
│   │   ├── engine.py        # Backtesting engine
│   │   ├── metrics.py       # Performance metrics
│   ├── strategies/
│   │   ├── base_strategy.py # Base class for strategies
│   │   ├── bollinger_bands.py
│   │   ├── macd_strategy.py
│   │   ├── moving_average.py
│   │   ├── rsi_strategy.py
│   │   ├── volume_momentum.py
```

## Getting Started
1. **Create and activate a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the backend**:
   ```powershell
   python main.py
   ```

## Adding Strategies
- Add new strategy modules in `app/strategies/` by subclassing `base_strategy.py`.

## Deployment
- Configuration for Cerebrium deployment is in `cerebrium.toml`.
- For Vercel integration, see `.vercel/project.json`.

## License
This project is licensed under the MIT License.
