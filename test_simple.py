import yfinance as yf
import pandas as pd

print("🚀 Testing basic stock data...")

# Test fetching Apple stock data
try:
    stock = yf.Ticker("META")
    data = stock.history(period="1d", interval="5m")
    
    if not data.empty:
        current_price = data['Close'].iloc[-1]
        print(f"✅ META current price: ${current_price:.2f}")
        print("✅ Basic functionality works!")
    else:
        print("❌ No data received")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("🎯 Test complete!")