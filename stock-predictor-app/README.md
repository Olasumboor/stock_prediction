# Stock Price Prediction App

This project is a Stock Price Prediction application that utilizes machine learning to forecast future stock prices based on historical data and technical indicators. The app is built using Streamlit, a popular framework for creating web applications in Python.

## Features

- Fetches real-time stock data using Yahoo Finance.
- Creates technical indicators such as moving averages and volatility.
- Trains a Random Forest machine learning model for predictions.
- Provides visualizations of historical and predicted stock prices.
- Displays model performance metrics and feature importance.
- Allows users to download prediction results in CSV format.

## Installation

To run this application, you need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stock-predictor-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   streamlit run stock_predictor.py
   ```

2. Open your web browser and go to `http://localhost:8501` to access the app.

3. Enter a stock ticker symbol (e.g., AAPL, MSFT) in the input field.

4. Select the historical data period for training and set the prediction settings.

5. Click on the "Run Prediction" button to generate predictions and view the results.

## Disclaimer

This tool is for educational purposes only. Stock market predictions are inherently uncertain, and this app should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct your own research.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.