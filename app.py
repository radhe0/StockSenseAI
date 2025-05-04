import streamlit as st
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from textblob import TextBlob
import requests
import json
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from alpha_vantage.fundamentaldata import FundamentalData
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from newsapi.newsapi_client import NewsApiClient

# Ensuring scaler is defined before usage
scaler = StandardScaler()

# API Configuration
ALPHA_VANTAGE_API_KEY = 'X9M080OGNUZXJYKW'  # Your Alpha Vantage key
MARKETSTACK_API_KEY = 'e0e0dd21b49091ac40772dde48146fbd'  # Your Marketstack key

# Page configuration
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .sidebar .sidebar-content {
            background-color: #1a1a2e;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00FFFF !important;
        }
        .stRadio > div {
            flex-direction: row !important;
            gap: 10px;
        }
        .stRadio [role="radiogroup"] {
            gap: 15px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .metric-card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .prediction-day {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            border-left: 4px solid #4CAF50;
        }
        .time-input {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .market-indicator {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .nse-indicator {
            background-color: #1f77b4;
            color: white;
        }
        .bse-indicator {
            background-color: #ff7f0e;
            color: white;
        }
        .us-indicator {
            background-color: #2ca02c;
            color: white;
        }
        .warning-box {
            background-color: #ffcccb;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.title('📈 StockSense AI')
st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;">
        <h3 style="color: #00FFFF; margin: 0;">Advanced Stock Price Prediction & Analysis</h3>
        <p style="color: #CCCCCC;">Predict future stock prices (US & Indian markets) with machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Indian stock symbols database (sample)
INDIAN_STOCKS = {
    'NSE': {
        'RELIANCE': 'RELIANCE.NS',
        'TATASTEEL': 'TATASTEEL.NS',
        'HDFCBANK': 'HDFCBANK.NS',
        'INFY': 'INFY.NS',
        'TCS': 'TCS.NS',
        'BHARTIARTL': 'BHARTIARTL.NS',
        'ITC': 'ITC.NS',
        'LT': 'LT.NS',
        'SBIN': 'SBIN.NS',
        'HINDUNILVR': 'HINDUNILVR.NS'
    },
    'BSE': {
        'RELIANCE': 'RELIANCE.BO',
        'TATASTEEL': 'TATASTEEL.BO',
        'HDFCBANK': 'HDFCBANK.BO',
        'INFY': 'INFY.BO',
        'TCS': 'TCS.BO'
    },
    'INDICES': {
        'NIFTY_50': '^NSEI',
        'BANK_NIFTY': '^NSEBANK',
        'NIFTY_IT': '^CNXIT',
        'NIFTY_AUTO': '^CNXAUTO',
        'NIFTY_FIN_SERVICE': '^CNXFINANCE',
        'NIFTY_FMCG': '^CNXFMCG',
        'NIFTY_METAL': '^CNXMETAL',
        'NIFTY_PHARMA': '^CNXPHARMA',
        'NIFTY_PSU_BANK': '^CNXPSUBANK',
        'NIFTY_REALTY': '^CNXREALTY'
    },
    'FUTURES': {
        'NIFTY_50_FUT': 'NIFTY_FUT.NS',
        'BANK_NIFTY_FUT': 'BANKNIFTY_FUT.NS'
    },
    'OPTIONS': {
        'NIFTY_50_CE': 'NIFTY_CE.NS',
        'NIFTY_50_PE': 'NIFTY_PE.NS',
        'BANK_NIFTY_CE': 'BANKNIFTY_CE.NS',
        'BANK_NIFTY_PE': 'BANKNIFTY_PE.NS'
    }
}

# Sidebar configuration
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2813/2813893.png", width=80)
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

# Sidebar info
with st.sidebar.expander("ℹ️ About"):
    st.write("""
        **StockSense AI** provides:
        - Real-time stock data visualization
        - Technical indicator analysis
        - Daily and intraday price predictions
        - Support for US and Indian markets
        - Multiple model comparison
    """)
    st.markdown("Created and designed by Ravi Yadav")

def safe_format(value, format_str=".2f"):
    """Safely format a value, handling None and non-numeric types"""
    try:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (int, float)):
            return format(value, format_str)
        return str(value)
    except:
        return str(value)

def safe_divide(numerator, denominator):
    """Safely divide two numbers, handling zero denominator"""
    try:
        if denominator == 0:
            return 0
        return numerator / denominator
    except:
        return 0

@st.cache_resource
def get_intraday_data(ticker, interval='1m', days=1):
    """Get intraday data for the given ticker"""
    end = datetime.datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            # Try with a longer period if no data found
            start = end - timedelta(days=7)
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        return df
    except Exception as e:
        st.error(f"Error downloading intraday data: {str(e)}")
        return pd.DataFrame()

def get_market_indicator(ticker):
    """Get market indicator for the ticker"""
    if ticker.endswith('.NS'):
        return '<span class="market-indicator nse-indicator">NSE</span>'
    elif ticker.endswith('.BO'):
        return '<span class="market-indicator bse-indicator">BSE</span>'
    else:
        return '<span class="market-indicator us-indicator">US</span>'

def get_news_sentiment(ticker):
    """Get news sentiment for the given ticker"""
    try:
        if not ticker or pd.isna(ticker):
            return None
            
        ticker_symbol = ticker.split('.')[0] if isinstance(ticker, str) else str(ticker)
        newsapi = NewsApiClient(api_key='15801f1eed47482b9affbcb0b5aa45f1')
        articles = newsapi.get_everything(q=ticker_symbol, language='en', sort_by='relevancy')
        
        sentiments = []
        for article in articles.get('articles', [])[:10]:  # Safely get articles
            title = article.get('title', '')
            description = article.get('description', '')
            
            # Skip if both title and description are empty
            if not title and not description:
                continue
                
            try:
                analysis = TextBlob(str(title) + ' ' + str(description))
                sentiments.append(analysis.sentiment.polarity)
            except Exception as e:
                continue
                
        if not sentiments:
            return None
            
        avg_sentiment = sum(sentiments) / len(sentiments)
        return {
            'average_sentiment': avg_sentiment,
            'articles_count': len(sentiments),
            'positive_articles': sum(1 for s in sentiments if s > 0),
            'negative_articles': sum(1 for s in sentiments if s < 0)
        }
    except Exception as e:
        st.error(f"Error fetching news sentiment: {str(e)}")
        return None


def get_fundamental_data(ticker):
    """Get fundamental data with proper derivative handling"""
    try:
        # Skip fundamental analysis for derivatives
        if any(x in ticker for x in ['_FUT', '_CE', '_PE']):
            return None
            
        stock = yf.Ticker(ticker)
        stats = stock.info
        
        # Handle cases where info is None
        if not stats:
            return None
            
        return {
            'pe_ratio': stats.get('trailingPE', 'N/A'),
            'forward_pe': stats.get('forwardPE', 'N/A'),
            'peg_ratio': stats.get('pegRatio', 'N/A'),
            'eps': stats.get('trailingEps', 'N/A'),
            'forward_eps': stats.get('forwardEps', 'N/A'),
            'dividend_yield': stats.get('dividendYield', 'N/A'),
            'profit_margin': stats.get('profitMargins', 'N/A'),
            'beta': stats.get('beta', 'N/A'),
            'market_cap': stats.get('marketCap', 'N/A'),
            'enterprise_value': stats.get('enterpriseValue', 'N/A'),
            'revenue_growth': stats.get('revenueGrowth', 'N/A'),
            'earnings_growth': stats.get('earningsGrowth', 'N/A'),
            'return_on_equity': stats.get('returnOnEquity', 'N/A'),
            'debt_to_equity': stats.get('debtToEquity', 'N/A')
        }
    except Exception as e:
        st.error(f"Error fetching fundamental data: {str(e)}")
        return None

def get_valid_expiries(symbol):
    """Get approximate expiry dates for the next 3 months"""
    try:
        today = datetime.date.today()
        # Generate approximate monthly expiries (3rd Friday of each month)
        expiries = []
        for i in range(3):  # Next 3 months
            dt = today + timedelta(days=30*i)
            # Find 3rd Friday of the month
            first_day = datetime.date(dt.year, dt.month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)
            expiries.append(third_friday)
        return sorted(expiries)
    except Exception:
        return []

def get_valid_strikes(symbol, expiry_date, option_type):
    """Generate common strike prices based on symbol"""
    try:
        if symbol == "NIFTY":
            current_price = 19500  # Approximate NIFTY price
            return sorted(list(range(current_price-1000, current_price+1000, 50)))
        elif symbol == "BANKNIFTY":
            current_price = 43000  # Approximate BANKNIFTY price
            return sorted(list(range(current_price-2000, current_price+2000, 100)))
        else:
            return list(range(10000, 50000, 100))
    except Exception:
        return []

def fetch_derivative_data(symbol, start_date, end_date, is_option=False, option_type=None, strike_price=None, expiry_date=None):
    """Fetch derivative data with better handling for Indian markets"""
    try:
        # For Indian derivatives, we'll use the underlying index data as proxy
        if symbol in ["NIFTY", "BANKNIFTY"]:
            # Get the underlying index data
            underlying_ticker = "^NSEI" if symbol == "NIFTY" else "^NSEBANK"
            data = yf.download(underlying_ticker, start=start_date, end=end_date)
            
            if not data.empty:
                # Add some simulated volatility to make it look like derivatives
                if is_option:
                    multiplier = 1.1 if option_type == "CE" else 0.9
                    data['Close'] = data['Close'] * multiplier
                return data
        
        # For other cases, try MarketStack first
        params = {
            'access_key': MARKETSTACK_API_KEY,
            'symbols': f"{symbol}{'_FUT' if not is_option else ''}",
            'date_from': start_date.strftime('%Y-%m-%d'),
            'date_to': end_date.strftime('%Y-%m-%d'),
            'limit': 1000
        }
        response = requests.get('http://api.marketstack.com/v1/eod', params=params)
        if response.status_code == 200:
            data = pd.DataFrame(response.json()['data'])
            if not data.empty:
                return data
        
        # Final fallback to underlying stock data
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
        
    except Exception as e:
        st.error(f"Error fetching derivative data: {str(e)}")
        return pd.DataFrame()

def get_common_strikes(symbol):
    # Provide a list of common strikes for NIFTY and BANKNIFTY
    if symbol == "NIFTY":
        return list(range(18000, 20500, 50))
    elif symbol == "BANKNIFTY":
        return list(range(40000, 46000, 100))
    else:
        return list(range(10000, 50000, 100))

def main():
    option = st.sidebar.selectbox('Select Mode', ['📊 Visualize', '📋 Recent Data', '🔮 Predict'])
    st.sidebar.markdown("---")
    
    market_category = st.sidebar.selectbox(
        'Select Category',
        options=['US', 'NSE', 'BSE', 'INDICES', 'FUTURES', 'OPTIONS'],
        index=0
    )
    
    if market_category == 'US':
        default_ticker = 'SPY'
        option_ticker = st.sidebar.text_input('Enter Stock Symbol', value=default_ticker).upper()
        data_fetch_method = 'yfinance'
    else:
        exchange_stocks = INDIAN_STOCKS.get(market_category, {})
        if exchange_stocks:
            selected_stock = st.sidebar.selectbox(
                'Select Stock',
                options=list(exchange_stocks.keys()),
                format_func=lambda x: f"{x} ({exchange_stocks[x]})"
            )
            option_ticker = exchange_stocks[selected_stock]
            data_fetch_method = 'yfinance'
            # For derivatives, switch fetch method
            if market_category in ['FUTURES', 'OPTIONS']:
                data_fetch_method = 'nsepy'
        else:
            st.sidebar.warning("No stocks/options available for this category.")
            option_ticker = None
            data_fetch_method = None

    st.sidebar.markdown("---")
    today = datetime.date.today()
    duration = st.sidebar.slider('Select Duration (days)', min_value=30, max_value=365*5, value=1000)
    before = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=before)
    end_date = st.sidebar.date_input('End Date', today)

    if st.sidebar.button('Apply Settings', key='apply_settings'):
        if start_date < end_date:
            st.sidebar.success(f'Data range: {start_date} to {end_date}')
        else:
            st.sidebar.error('Error: End date must be after start date')

    st.sidebar.markdown("---")

    # --- NSEpy integration for derivatives ---
    data = pd.DataFrame()
    # In the main() function, replace the data fetching part with:
    if data_fetch_method == 'yfinance':
        try:
            data = yf.download(option_ticker, start=start_date, end=end_date)
            if data.empty:
                st.warning("No data found for this stock symbol. Please try another.")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            data = pd.DataFrame()
    elif data_fetch_method == 'nsepy' and option_ticker:
        symbol = "NIFTY" if "NIFTY" in selected_stock else "BANKNIFTY"
        expiry_date = st.sidebar.date_input('Expiry Date', value=end_date)
        
        if market_category == 'OPTIONS':
            option_type = "CE" if "CE" in selected_stock else "PE"
            common_strikes = get_common_strikes(symbol)
            strike_price = st.sidebar.selectbox('Strike Price', options=common_strikes)
        
        if end_date > expiry_date:
            st.sidebar.warning("End date cannot be after expiry date.")
            data = pd.DataFrame()
        else:
            data = fetch_derivative_data(
                symbol, 
                start_date, 
                end_date, 
                market_category == 'OPTIONS',
                option_type if market_category == 'OPTIONS' else None,
                strike_price if market_category == 'OPTIONS' else None,
                expiry_date
            )
            if data.empty:
                st.warning("Using underlying index data as proxy for derivatives")
        st.sidebar.info("Using simulated derivatives data based on underlying indices")
    # --- End NSEpy integration ---

    if data.empty:
        st.warning("No data available for the selected stock symbol and date range.")
        return

    if option == '📊 Visualize':
        tech_indicators(data, option_ticker)
    elif option == '📋 Recent Data':
        dataframe(data, option_ticker)
    else:
        predict(option_ticker, data)

@st.cache_resource
def download_data(op, start_date, end_date):
    try:
        df = yf.download(op, start=start_date, end=end_date, progress=False)
        if df.empty:
            # Try with auto_adjust=True for Indian stocks
            if op.endswith(('.NS', '.BO')):
                df = yf.download(op, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if df.empty:
                st.error("No data found for this stock symbol. Please try another.")
                return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return pd.DataFrame()

def tech_indicators(data, ticker):
    st.header('Technical Indicators Analysis')
    st.markdown(f"**Stock:** {ticker} {get_market_indicator(ticker)}", unsafe_allow_html=True)
    
    if data.empty:
        st.warning("No data available for visualization.")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        option = st.radio('Select Indicator', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'], horizontal=True)
    
    with col2:
        window_size = st.slider('Window Size', min_value=5, max_value=50, value=14, key='window_size')
    
    close_prices = pd.Series(data['Close'].values.flatten(), index=data.index)
    
    fig = make_subplots(rows=1, cols=1)
    
    # Common layout settings for all charts
    layout_settings = {
        'template': 'plotly_dark',
        'hovermode': 'x unified',
        'height': 600,
        'margin': dict(l=50, r=50, b=100, t=50, pad=4),
        'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        'xaxis': {
            'type': 'date',
            'tickformat': '%b %Y',
            'tickmode': 'auto',
            'nticks': 12,
            'tickangle': 45,
            'showgrid': True,
            'ticklabelmode': 'period'
        }
    }
    
    if option == 'Close':
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=close_prices, 
            name='Close Price', 
            line=dict(color='#00FFFF')
        ))
        fig.update_layout(
            title='Close Price',
            yaxis_title='Price',
            **layout_settings
        )
    
    elif option == 'BB':
        bb_indicator = BollingerBands(close=close_prices, window=window_size)
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=close_prices, 
            name='Close Price', 
            line=dict(color='#00FFFF')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=bb_indicator.bollinger_hband(), 
            name='Upper Band', 
            line=dict(color='#FF6347', dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=bb_indicator.bollinger_lband(), 
            name='Lower Band', 
            line=dict(color='#32CD32', dash='dot')
        ))
        fig.update_layout(
            title='Bollinger Bands',
            yaxis_title='Price',
            **layout_settings
        )
    
    elif option == 'MACD':
        macd_line = MACD(close=close_prices, window_slow=26, window_fast=12).macd()
        signal_line = MACD(close=close_prices, window_slow=26, window_fast=12).macd_signal()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=macd_line, 
            name='MACD Line', 
            line=dict(color='#00FFFF')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=signal_line, 
            name='Signal Line', 
            line=dict(color='#FFA500')
        ))
        fig.update_layout(
            title='MACD (Moving Average Convergence Divergence)',
            yaxis_title='Value',
            **layout_settings
        )
    
    elif option == 'RSI':
        rsi = RSIIndicator(close=close_prices, window=window_size).rsi()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=rsi, 
            name='RSI', 
            line=dict(color='#00FFFF')
        ))
        fig.update_layout(
            title='Relative Strength Index (RSI)',
            yaxis_title='RSI Value',
            **layout_settings
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    elif option == 'SMA':
        sma = SMAIndicator(close=close_prices, window=window_size).sma_indicator()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=close_prices, 
            name='Close Price', 
            line=dict(color='#00FFFF')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=sma, 
            name=f'SMA {window_size}', 
            line=dict(color='#FFA500')
        ))
        fig.update_layout(
            title='Simple Moving Average',
            yaxis_title='Price',
            **layout_settings
        )
    
    else:  # EMA
        ema = EMAIndicator(close=close_prices, window=window_size).ema_indicator()
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=close_prices, 
            name='Close Price', 
            line=dict(color='#00FFFF')
        ))
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=ema, 
            name=f'EMA {window_size}', 
            line=dict(color='#FFA500')
        ))
        fig.update_layout(
            title='Exponential Moving Average',
            yaxis_title='Price',
            **layout_settings
        )
    
    st.plotly_chart(fig, use_container_width=True)

def dataframe(data, ticker):
    st.header('Recent Market Data')
    st.markdown(f"**Stock:** {ticker} {get_market_indicator(ticker)}", unsafe_allow_html=True)
    
    if data.empty:
        st.warning("No data available to display.")
        return
    
    columns_to_show = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Adj Close' in data.columns:
        columns_to_show.append('Adj Close')
    
    def color_negative_red(val):
        if isinstance(val, (int, float)):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'
        return ''
    
    display_data = data[columns_to_show].tail(10)
    numeric_cols = display_data.select_dtypes(include=['float64', 'int64']).columns
    styled_data = display_data.style.format("{:.2f}").applymap(color_negative_red, subset=numeric_cols)
    
    st.dataframe(styled_data, height=400, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_price = data['Close'].iloc[-1] if len(data) > 0 else None
        st.markdown(f"""
            <div class="metric-card">
                <h4>Current Price</h4>
                <h3>{safe_format(current_price)}</h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if len(data) > 1:
            prev_close = data['Close'].iloc[-2].item() if hasattr(data['Close'].iloc[-2], 'item') else data['Close'].iloc[-2]
            current_close = data['Close'].iloc[-1].item() if hasattr(data['Close'].iloc[-1], 'item') else data['Close'].iloc[-1]
            
            change = current_close - prev_close
            percent_change = safe_divide(change * 100, prev_close)
            
            color = 'red' if change < 0 else 'green'
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Daily Change</h4>
                    <h3 style="color: {color}">
                        {safe_format(change)} ({safe_format(percent_change)}%)
                    </h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Daily Change</h4>
                    <h3>N/A</h3>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        volume = data['Volume'].iloc[-1] if len(data) > 0 else None
        st.markdown(f"""
            <div class="metric-card">
                <h4>Volume (Latest)</h4>
                <h3>{safe_format(volume, ",.0f") if volume is not None else 'N/A'}</h3>
            </div>
        """, unsafe_allow_html=True)

class StockPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=50, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(50, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        dropout = self.dropout(last_time_step)
        predictions = self.linear(dropout)
        return predictions

def predict(ticker, daily_data):
    st.header('Stock Price Prediction')
    st.markdown(f"**Stock:** {ticker} {get_market_indicator(ticker)}", unsafe_allow_html=True)
    
    if daily_data.empty:
        st.warning("No data available for prediction.")
        return
    
    # Add sentiment and fundamental analysis sections
    with st.expander("📰 News Sentiment Analysis"):
        sentiment = get_news_sentiment(ticker)
        if sentiment:
            st.metric("Average Sentiment", f"{sentiment['average_sentiment']:.2f}", 
                     delta=f"{sentiment['positive_articles']} positive / {sentiment['negative_articles']} negative articles")
            st.progress((sentiment['average_sentiment'] + 1) / 2)
        else:
            st.warning("Could not fetch news sentiment data")
    
    with st.expander("📊 Fundamental Analysis"):
        if any(x in ticker for x in ['_FUT', '_CE', '_PE']):
            st.info("Fundamental analysis not available for derivative contracts")
        else:
            fundamentals = get_fundamental_data(ticker)
            if fundamentals:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("P/E Ratio", safe_format(fundamentals['pe_ratio']))
                    st.metric("Forward P/E", safe_format(fundamentals['forward_pe']))
                    st.metric("EPS", safe_format(fundamentals['eps']))
                    st.metric("Forward EPS", safe_format(fundamentals['forward_eps']))
                    
                with col2:
                    st.metric("Dividend Yield", f"{safe_format(fundamentals['dividend_yield'], '.2%') if isinstance(fundamentals['dividend_yield'], (int, float)) else 'N/A'}")
                    st.metric("Profit Margin", f"{safe_format(fundamentals['profit_margin'], '.2%') if isinstance(fundamentals['profit_margin'], (int, float)) else 'N/A'}")
                    st.metric("Beta", safe_format(fundamentals['beta']))
                    st.metric("PEG Ratio", safe_format(fundamentals['peg_ratio']))
                    
                with col3:
                    st.metric("Market Cap", f"${safe_format(fundamentals['market_cap'], ',.0f') if isinstance(fundamentals['market_cap'], (int, float)) else 'N/A'}")
                    st.metric("Enterprise Value", f"${safe_format(fundamentals['enterprise_value'], ',.0f') if isinstance(fundamentals['enterprise_value'], (int, float)) else 'N/A'}")
                    st.metric("ROE", f"{safe_format(fundamentals['return_on_equity'], '.2%') if isinstance(fundamentals['return_on_equity'], (int, float)) else 'N/A'}")
                    st.metric("Debt/Equity", safe_format(fundamentals['debt_to_equity']))
            else:
                st.warning("Could not fetch fundamental data")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model = st.selectbox('Select Model', 
                           ['Linear Regression', 'Random Forest', 'Extra Trees', 'K-Neighbors', 'XGBoost', 'PyTorch LSTM'],
                           index=1)
        
        prediction_type = st.radio('Prediction Type', ['Daily', 'Intraday'], horizontal=True)
        
        if prediction_type == 'Daily':
            num = st.slider('Forecast Days', min_value=1, max_value=30, value=5)
        else:
            # Ensure this block is properly indented under the else clause
            if ticker.endswith(('.NS', '.BO')):
                st.markdown("""
                    <div class="warning-box">
                        Note: Intraday predictions for Indian stocks may be less accurate due to data limitations.
                        For best results, use during Indian market hours (9:15 AM to 3:30 PM IST).
                    </div>
                """, unsafe_allow_html=True)
                
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    hours = st.slider('Hours', min_value=0, max_value=23, value=0)
                with col1_2:
                    minutes = st.slider('Minutes', min_value=1, max_value=59, value=30, step=1)
                
                total_minutes = hours * 60 + minutes
                st.caption(f"Predicting for {hours}h {minutes}m in the future")
            else:
                # Add similar controls for non-Indian stocks
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    hours = st.slider('Hours', min_value=0, max_value=23, value=0)
                with col1_2:
                    minutes = st.slider('Minutes', min_value=1, max_value=59, value=30, step=1)
                
                total_minutes = hours * 60 + minutes
                st.caption(f"Predicting for {hours}h {minutes}m in the future")
    
        if st.button('Run Prediction', key='predict_button'):
            with st.spinner('Training model and making predictions...'):
                if model == 'Linear Regression':
                    engine = LinearRegression()
                elif model == 'Random Forest':
                    engine = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model == 'Extra Trees':
                    engine = ExtraTreesRegressor(n_estimators=100, random_state=42)
                elif model == 'K-Neighbors':
                    engine = KNeighborsRegressor(n_neighbors=5)
                elif model == 'PyTorch LSTM':
                    # Convert data to PyTorch tensors
                    df = daily_data[['Close']]
                    df['preds'] = daily_data.Close.shift(-num)
                    x = df.drop(['preds'], axis=1).values
                    x = scaler.fit_transform(x)
                    x_forecast = x[-num:]
                    x = x[:-num]
                    y = df.preds.values
                    y = y[:-num]
                    
                    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
                    
                    # Reshape for LSTM [samples, timesteps, features]
                    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                    
                    X_train_tensor = torch.FloatTensor(X_train)
                    y_train_tensor = torch.FloatTensor(y_train)
                    X_test_tensor = torch.FloatTensor(X_test)
                    
                    # Create DataLoader
                    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
                    
                    # Initialize model
                    model = StockPredictor(input_size=X_train.shape[2])
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    
                    # Training loop
                    for epoch in range(100):
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                    
                    # Evaluation
                    with torch.no_grad():
                        model.eval()
                        test_preds = model(X_test_tensor).numpy()
                        mae = mean_absolute_error(y_test, test_preds)
                        
                    # Make predictions
                    x_forecast = x_forecast.reshape((x_forecast.shape[0], 1, x_forecast.shape[1]))
                    x_forecast_tensor = torch.FloatTensor(x_forecast)
                    with torch.no_grad():
                        model.eval()
                        forecast_pred = model(x_forecast_tensor).numpy().flatten()
                    
                    # Display results
                    last_date = daily_data.index[-1]
                    prediction_dates = [last_date + datetime.timedelta(days=i) for i in range(1, num+1)]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4>Model Performance</h4>
                                <p><strong>MAE:</strong> {safe_format(mae, ".4f")}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4>{num}-Day Price Forecast</h4>
                        """, unsafe_allow_html=True)
                        
                        for i, (date, pred) in enumerate(zip(prediction_dates, forecast_pred)):
                            st.markdown(f"""
                                <div class="prediction-day">
                                    <strong>Day {i+1}</strong> ({date.strftime('%Y-%m-%d')}): ${safe_format(pred)}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=daily_data.index[-30:],
                        y=daily_data['Close'].values[-30:],
                        name='Actual Prices',
                        line=dict(color='#00FFFF', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=prediction_dates,
                        y=forecast_pred,
                        name='Predicted Prices',
                        line=dict(color='#FFA500', width=2, dash='dot')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=prediction_dates,
                        y=forecast_pred,
                        mode='markers',
                        name='Prediction Points',
                        marker=dict(color='#FFA500', size=8)
                    ))
                    
                    fig.update_layout(
                        title='Actual vs Predicted Prices',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        template='plotly_dark',
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(
                            type='date',
                            tickformat='%b %Y',
                            tickmode='auto',
                            nticks=12,
                            tickangle=45,
                            showgrid=True,
                            ticklabelmode='period'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    return
                else:
                    engine = XGBRegressor(n_estimators=100, random_state=42)
                    
                if prediction_type == 'Daily':
                    daily_model_engine(engine, num, daily_data, ticker)
                else:
                    intraday_model_engine(engine, ticker, total_minutes)

def daily_model_engine(model, num, data, ticker):
    scaler = StandardScaler()
    
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    x_forecast = x[-num:]
    x = x[:-num]
    y = df.preds.values
    y = y[:-num]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h4>Model Performance</h4>
                <p><strong>R² Score:</strong> {safe_format(r2, ".4f")}</p>
                <p><strong>MAE:</strong> {safe_format(mae, ".4f")}</p>
            </div>
        """, unsafe_allow_html=True)
    
    forecast_pred = model.predict(x_forecast)
    last_date = data.index[-1]
    prediction_dates = [last_date + datetime.timedelta(days=i) for i in range(1, num+1)]
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h4>{num}-Day Price Forecast</h4>
        """, unsafe_allow_html=True)
        
        for i, (date, pred) in enumerate(zip(prediction_dates, forecast_pred)):
            st.markdown(f"""
                <div class="prediction-day">
                    <strong>Day {i+1}</strong> ({date.strftime('%Y-%m-%d')}): ${safe_format(pred)}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index[-30:],
        y=data['Close'].values[-30:],
        name='Actual Prices',
        line=dict(color='#00FFFF', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction_dates,
        y=forecast_pred,
        name='Predicted Prices',
        line=dict(color='#FFA500', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction_dates,
        y=forecast_pred,
        mode='markers',
        name='Prediction Points',
        marker=dict(color='#FFA500', size=8)
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            type='date',
            tickformat='%b %Y',
            tickmode='auto',
            nticks=12,
            tickangle=45,
            showgrid=True,
            ticklabelmode='period'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def intraday_model_engine(model, ticker, minutes_to_predict):
    # Get recent intraday data
    intraday_data = get_intraday_data(ticker, interval='1m', days=7)
    
    if intraday_data.empty:
        if ticker.endswith(('.NS', '.BO')):
            st.error("Could not fetch intraday data. For Indian stocks, please try during market hours (9:15 AM to 3:30 PM IST).")
        else:
            st.error("Could not fetch intraday data for prediction. Please try again during market hours.")
        return
    
    if len(intraday_data) < minutes_to_predict * 2:
        st.warning(f"Not enough intraday data available. Need at least {minutes_to_predict*2} minutes of data.")
        return
    
    scaler = StandardScaler()
    
    # Prepare data for minute-level prediction
    df = intraday_data[['Close']]
    df['preds'] = df.Close.shift(-minutes_to_predict)
    df = df.dropna()
    
    if len(df) < minutes_to_predict:
        st.warning("Not enough data points for the requested prediction horizon.")
        return
    
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    y = df.preds.values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    # Predict future minutes
    last_points = x[-minutes_to_predict:]
    forecast_pred = model.predict(last_points)
    last_time = df.index[-1]
    
    # Generate prediction times
    prediction_times = [last_time + timedelta(minutes=i+1) for i in range(minutes_to_predict)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h4>Model Performance</h4>
                <p><strong>R² Score:</strong> {safe_format(r2, ".4f")}</p>
                <p><strong>MAE:</strong> {safe_format(mae, ".4f")}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hours = minutes_to_predict // 60
        mins = minutes_to_predict % 60
        st.markdown(f"""
            <div class="metric-card">
                <h4>Next {hours}h {mins}m Price Forecast</h4>
        """, unsafe_allow_html=True)
        
        for i, (time, pred) in enumerate(zip(prediction_times, forecast_pred)):
            st.markdown(f"""
                <div class="prediction-day">
                    <strong>{time.strftime('%H:%M')}</strong>: ${safe_format(pred)}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Plot the results
    fig = go.Figure()
    
    # Show last 60 minutes of actual data
    plot_data = df.iloc[-60:]
    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data['Close'],
        name='Actual Prices',
        line=dict(color='#00FFFF', width=2)
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=prediction_times,
        y=forecast_pred,
        name='Predicted Prices',
        line=dict(color='#FFA500', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction_times,
        y=forecast_pred,
        mode='markers',
        name='Prediction Points',
        marker=dict(color='#FFA500', size=8)
    ))
    
    fig.update_layout(
        title=f'Actual vs Predicted Prices (Next {hours}h {mins}m)',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_dark',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            type='date',
            tickformat='%H:%M',
            tickmode='auto',
            nticks=12,
            showgrid=True
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
