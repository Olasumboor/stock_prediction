import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def get_stock_data(ticker, start_date=None, end_date=None, period='1y'):
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date)
        else:
            data = yf.download(ticker, period=period)

        if data.empty:
            return None, f"No data found for ticker: {ticker}"

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        data['Date'] = data.index
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['DayOfWeek'] = data['Date'].dt.dayofweek

        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()

        data['Price_Change'] = data['Close'] - data['Open']
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()

        return data, f"Successfully downloaded data for {ticker}"

    except Exception as e:
        return None, f"Error: {str(e)}"

def prepare_prediction_data(data, forecast_period=1):
    data_copy = data.copy()
    data_copy['Target'] = data_copy['Close'].shift(-forecast_period)

    for lag in [1, 2, 3, 5]:
        data_copy[f'Close_lag_{lag}'] = data_copy['Close'].shift(lag)
        data_copy[f'Volume_lag_{lag}'] = data_copy['Volume'].shift(lag)

    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA20', 'MA50', 'Returns', 'Volatility',
        'Price_Change', 'High_Low_Ratio', 'Volume_MA',
        'Year', 'Month', 'Day', 'DayOfWeek'
    ]

    for lag in [1, 2, 3, 5]:
        feature_columns.extend([f'Close_lag_{lag}', f'Volume_lag_{lag}'])

    available_features = [col for col in feature_columns if col in data_copy.columns]
    data_clean = data_copy.dropna()

    if len(data_clean) == 0:
        raise ValueError("No clean data available after removing NaN values")

    X = data_clean[available_features]
    y = data_clean['Target']

    return X, y, data_clean

class StockPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def train(self, X, y):
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        self.is_trained = True
        return metrics
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_future(self, data, days_ahead=10):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        predictions = []
        prediction_dates = []
        
        current_data = data.copy()
        last_date = current_data.index[-1]
        
        for day in range(days_ahead):
            latest_features = current_data.iloc[[-1]][self.feature_names]
            next_price = self.predict(latest_features)[0]
            predictions.append(next_price)
            
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)
            prediction_dates.append(next_date)
            last_date = next_date
            
            new_row = current_data.iloc[-1].copy()
            new_row['Close'] = next_price
            new_row['Open'] = next_price * (1 + np.random.normal(0, 0.01))
            new_row['High'] = max(new_row['Open'], next_price) * (1 + abs(np.random.normal(0, 0.005)))
            new_row['Low'] = min(new_row['Open'], next_price) * (1 - abs(np.random.normal(0, 0.005)))
            new_row['Volume'] = current_data['Volume'].iloc[-1]
            
            new_row['Year'] = next_date.year
            new_row['Month'] = next_date.month
            new_row['Day'] = next_date.day
            new_row['DayOfWeek'] = next_date.weekday()
            
            if len(current_data) >= 5:
                new_row['MA5'] = current_data['Close'].iloc[-4:].tolist() + [next_price]
                new_row['MA5'] = np.mean(new_row['MA5'])
            
            current_data.loc[next_date] = new_row
        
        return predictions, prediction_dates

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("🚀 Stock Price Prediction App")
st.markdown("""
This app uses machine learning (Random Forest) to predict future stock prices based on historical data and technical indicators.
""")

st.sidebar.header("📊 Configuration")

ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL", help="Enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL)")

st.sidebar.subheader("📅 Historical Data Period")
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}
selected_period = st.sidebar.selectbox("Select Period", list(period_options.keys()), index=3)
period = period_options[selected_period]

st.sidebar.subheader("🔮 Prediction Settings")
days_to_predict = st.sidebar.slider("Days to Predict", 1, 30, 10)
forecast_period = st.sidebar.selectbox("Forecast Period (training)", [1, 5, 10], index=0, 
                                     help="Number of days ahead the model is trained to predict")

run_prediction = st.sidebar.button("🚀 Run Prediction", type="primary")

if run_prediction:
    if not ticker:
        st.error("Please enter a ticker symbol!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📥 Fetching stock data...")
        progress_bar.progress(20)
        
        data, message = get_stock_data(ticker.upper(), period=period)
        
        if data is None:
            st.error(f"❌ {message}")
        else:
            st.success(f"✅ {message}")
            
            status_text.text("🔧 Preparing data for training...")
            progress_bar.progress(40)
            
            try:
                X, y, processed_data = prepare_prediction_data(data, forecast_period)
                
                if len(X) < 50:
                    st.warning("⚠️ Limited data available. Consider using a longer time period for better predictions.")
                
                status_text.text("🤖 Training machine learning model...")
                progress_bar.progress(60)
                
                predictor = StockPricePredictor()
                metrics = predictor.train(X, y)
                
                status_text.text("🔮 Generating predictions...")
                progress_bar.progress(80)
                
                future_predictions, prediction_dates = predictor.predict_future(
                    processed_data, days_ahead=days_to_predict
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📈 Price Prediction Chart")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    hist_data = data.tail(min(60, len(data)))
                    ax.plot(hist_data.index, hist_data['Close'], 
                           label='Historical Price', color='blue', linewidth=2)
                    
                    ax.plot(prediction_dates, future_predictions, 
                           label='Predicted Price', color='red', linewidth=2, marker='o', markersize=4)
                    
                    ax.axvline(x=hist_data.index[-1], color='green', linestyle='--', alpha=0.7, label='Today')
                    
                    ax.set_title(f"{ticker.upper()} Stock Price Prediction", fontsize=16, fontweight='bold')
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Price ($)', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=10)
                    
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📊 Model Performance")
                    st.metric("Test R² Score", f"{metrics['test_r2']:.3f}")
                    st.metric("Test RMSE", f"${metrics['test_rmse']:.2f}")
                    st.metric("Test MAE", f"${metrics['test_mae']:.2f}")
                    
                    st.subheader("💰 Price Summary")
                    current_price = data['Close'].iloc[-1]
                    predicted_price = future_predictions[-1]
                    price_change = predicted_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric(f"Predicted Price ({days_to_predict}d)", 
                             f"${predicted_price:.2f}", 
                             f"{price_change:+.2f} ({price_change_pct:+.1f}%)")
                
                st.subheader("📋 Detailed Predictions")
                predictions_df = pd.DataFrame({
                    'Date': prediction_dates,
                    'Predicted_Price': [f"${p:.2f}" for p in future_predictions]
                })
                st.dataframe(predictions_df, use_container_width=True)
                
                csv = pd.DataFrame({
                    'Date': prediction_dates,
                    'Predicted_Price': future_predictions
                }).to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"{ticker.upper()}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("🔍 Model Insights"):
                    st.write("**Feature Importance (Top 10):**")
                    feature_importance = pd.DataFrame({
                        'Feature': predictor.feature_names,
                        'Importance': predictor.model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    st.bar_chart(feature_importance.set_index('Feature'))
                    
                    st.write("**Training Metrics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training R²", f"{metrics['train_r2']:.3f}")
                    with col2:
                        st.metric("Training RMSE", f"${metrics['train_rmse']:.2f}")
                    with col3:
                        st.metric("Training MAE", f"${metrics['train_mae']:.2f}")
                
            except Exception as e:
                st.error(f"❌ Error in prediction process: {str(e)}")

else:
    st.markdown("""
    ## 🎯 How to Use This App
    
    1. **Enter a stock ticker** (e.g., AAPL, MSFT, GOOGL, TSLA)
    2. **Select the historical data period** for training
    3. **Choose prediction settings** (days to predict, forecast period)
    4. **Click 'Run Prediction'** to train the model and see results
    
    ## 🧠 What This App Does
    
    - Fetches real-time stock data using Yahoo Finance
    - Creates technical indicators (moving averages, volatility, etc.)
    - Trains a Random Forest machine learning model
    - Predicts future stock prices with confidence metrics
    - Provides downloadable results and model insights
    
    ## ⚠️ Disclaimer
    
    **This tool is for educational purposes only.** Stock market predictions are inherently uncertain, 
    and this app should not be used as the sole basis for investment decisions. Always consult with 
    financial professionals and do your own research.
    """)
    
    st.markdown("### 💡 Popular Tickers to Try:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**Tech**\nAAPL, MSFT, GOOGL")
    with col2:
        st.info("**Finance**\nJPM, BAC, GS")
    with col3:
        st.info("**Auto**\nTSLA, F, GM")
    with col4:
        st.info("**Retail**\nAMZN, WMT, TGT")