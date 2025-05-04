# StockSense AI - Advanced Stock Prediction & Analysis Platform

![StockSense AI Banner](https://via.placeholder.com/1200x400?text=StockSense+AI+Banner)

StockSense AI is a powerful stock market analysis and prediction platform that combines machine learning models with technical indicators to provide actionable insights for traders and investors. The application supports both US and Indian markets, including equities, indices, futures, and options.

## ✨ Features

- **Multi-Market Support**: Analyze US stocks, Indian NSE/BSE equities, indices, futures, and options
- **Technical Analysis**: Interactive visualization of Bollinger Bands, MACD, RSI, SMA, EMA and more
- **Price Prediction**: Multiple ML models (Linear Regression, Random Forest, XGBoost, LSTM) for daily and intraday predictions
- **Derivatives Analysis**: Specialized tools for futures and options trading
- **News Sentiment**: Real-time news sentiment analysis for stocks
- **Fundamental Data**: Key financial ratios and metrics
- **Beautiful Visualization**: Interactive Plotly charts with dark mode support

## Installation
```python
git clone https://github.com/yourusername/StockSense-AI.git
cd StockSense-AI

```
2 .Create and activate a virtual environment:
```python
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. 3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
streamlit run app.py

```
The application will open in your default browser at http://localhost:8501


### note: 
- You need to have an API key from Alpha Vantage, News API, and Polygon for the application to work.
1. ALPHA_VANTAGE_API_KEY=your_key_here
2. MARKETSTACK_API_KEY=your_key_here
3. NEWSAPI_API_KEY=your_key_here


## Application Workflow:
1. Select market category (US, NSE, BSE, INDICES, FUTURES, OPTIONS)
2. Choose stock/derivative
3. Set date range
4. Select analysis mode:
  - 📊 Visualize: Technical indicators
  - 📋 Recent Data: Raw price data
  - 🔮 Predict: Price forecasting


## 📊 Technical Indicators
StockSense AI supports the following technical indicators:

- Bollinger Bands
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Average True Range (ATR)
- On-Balance Volume (OBV)


## 🤖 Machine Learning Models
The prediction module uses several ML algorithms:

- Linear Regression
- Random Forest
- XGBoost
- Extra Trees
- K-Neighbors
- LSTM Neural Networks (PyTorch implementation)

## 📈 Supported Markets
### US Markets
- All major US stocks and ETFs (via Yahoo Finance)
- Example: AAPL, MSFT, SPY, QQQ

### Indian Markets
Equities:

- NSE: RELIANCE, TATASTEEL, HDFCBANK, INFY, etc.
- BSE: RELIANCE.BO, TATASTEEL.BO, etc.

Indices:

- NIFTY 50, NIFTY NEXT 50, BANK NIFTY, etc.

Derivatives:

- NIFTY Futures & Options
- BANK NIFTY Futures & Options


## 📂 Project Structure
StockSense-AI/
├── app.py               # Main application file
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
├── style.css            # Custom CSS styles
└── venv/                # Virtual environment

## 📜 Requirements 
The project requires Python 3.8+ and the following packages:

- streamlit
- pandas
- yfinance
- ta
- scikit-learn
- xgboost
- plotly
- textblob
- alpha_vantage
- newsapi
- torch


## 🤝 Contributing  
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch ( git checkout -b feature/AmazingFeature )
3. Commit your changes ( git commit -m 'Add some AmazingFeature' )
4. Push to the branch ( git push origin feature/AmazingFeature )
5. Open a Pull Request

## 📄 License   
Distributed under the MIT License. See LICENSE for more information.

## 📧 Contact
Ravi Yadav - your.email@example.com

Project Link: https://github.com/yourusername/StockSense-AI

