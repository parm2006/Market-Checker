#!/usr/bin/env python3
"""
Financial Market Monitor with AI Sentiment Analysis
==================================================

A comprehensive tool for monitoring stock market movements, analyzing financial news sentiment,
and generating AI-powered market insights.

Features:
- Real-time stock data fetching (15-20min delay)
- Financial news sentiment analysis using FinBERT
- Price-sentiment correlation analysis  
- Interactive candlestick charts with sentiment overlays
- Automated market insights generation
- Streamlit dashboard interface

Author: Financial AI Assistant
Dependencies: See requirements section below
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import re

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# NLP and Sentiment Analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HF_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available. Install with: pip install transformers torch")
    HF_AVAILABLE = False

# News scraping
from bs4 import BeautifulSoup
import feedparser

# Dashboard (optional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

warnings.filterwarnings('ignore')

class NewsManager:
    """Handles fetching and processing financial news from multiple sources"""
    
    def __init__(self, newsapi_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self.news_sources = {
            'yahoo_rss': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch_rss': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'cnbc_rss': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114'
        }
    
    def fetch_rss_news(self, symbol: str = None, limit: int = 20) -> List[Dict]:
        """Fetch news from RSS feeds"""
        all_news = []
        
        for source_name, url in self.news_sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit//len(self.news_sources)]:
                    news_item = {
                        'title': entry.title,
                        'description': entry.get('summary', ''),
                        'url': entry.link,
                        'published': entry.get('published', ''),
                        'source': source_name
                    }
                    all_news.append(news_item)
            except Exception as e:
                print(f"Error fetching from {source_name}: {e}")
                continue
        
        # Filter for symbol-specific news if provided
        if symbol:
            filtered_news = []
            for news in all_news:
                text = f"{news['title']} {news['description']}".lower()
                if symbol.lower() in text or self._contains_stock_keywords(text):
                    filtered_news.append(news)
            return filtered_news
        
        return all_news[:limit]
    
    def fetch_newsapi_news(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Fetch news from NewsAPI (requires free API key)"""
        if not self.newsapi_key:
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{symbol} stock OR {symbol} earnings OR {symbol} financial",
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': limit,
            'apiKey': self.newsapi_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                return [{
                    'title': article['title'],
                    'description': article['description'] or '',
                    'url': article['url'],
                    'published': article['publishedAt'],
                    'source': 'newsapi'
                } for article in articles]
        except Exception as e:
            print(f"NewsAPI error: {e}")
        
        return []
    
    def _contains_stock_keywords(self, text: str) -> bool:
        """Check if text contains general stock market keywords"""
        keywords = ['stock', 'market', 'trading', 'earnings', 'financial', 'nasdaq', 'dow', 'sp500']
        return any(keyword in text for keyword in keywords)

class SentimentAnalyzer:
    """Handles sentiment analysis using FinBERT or fallback methods"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        if not HF_AVAILABLE:
            print("📊 Using rule-based sentiment analysis (transformers not available)")
            return
        
        try:
            model_name = "ProsusAI/finbert"
            print("🤖 Loading FinBERT model for financial sentiment analysis...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            print("✅ FinBERT model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load FinBERT: {e}")
            print("📊 Falling back to rule-based sentiment analysis")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of given text"""
        if self.sentiment_pipeline:
            return self._analyze_with_finbert(text)
        else:
            return self._analyze_rule_based(text)
    
    def _analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """Use FinBERT for sentiment analysis"""
        try:
            # Truncate text to avoid token limits
            text = text[:512]
            results = self.sentiment_pipeline(text)[0]
            
            sentiment_scores = {}
            for result in results:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = score
            
            # Convert to standardized format
            return {
                'positive': sentiment_scores.get('positive', 0),
                'negative': sentiment_scores.get('negative', 0),
                'neutral': sentiment_scores.get('neutral', 0),
                'compound': sentiment_scores.get('positive', 0) - sentiment_scores.get('negative', 0)
            }
        except Exception as e:
            print(f"FinBERT analysis error: {e}")
            return self._analyze_rule_based(text)
    
    def _analyze_rule_based(self, text: str) -> Dict[str, float]:
        """Fallback rule-based sentiment analysis"""
        text = text.lower()
        
        positive_words = [
            'gain', 'rise', 'up', 'surge', 'bull', 'positive', 'growth', 'profit',
            'beat', 'exceed', 'strong', 'robust', 'optimistic', 'upgrade', 'buy'
        ]
        
        negative_words = [
            'fall', 'drop', 'down', 'decline', 'bear', 'negative', 'loss', 'weak',
            'miss', 'disappoint', 'concern', 'risk', 'downgrade', 'sell', 'crash'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        total_words = len(text.split())
        
        if total_words == 0:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0}
        
        pos_score = pos_count / total_words
        neg_score = neg_count / total_words
        neutral_score = max(0, 1 - pos_score - neg_score)
        compound = pos_score - neg_score
        
        return {
            'positive': pos_score,
            'negative': neg_score,
            'neutral': neutral_score,
            'compound': compound
        }

class StockDataManager:
    """Handles stock data fetching and processing"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_stock_data(self, symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        """Fetch stock data with caching"""
        cache_key = f"{symbol}_{period}_{interval}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                print(f"⚠️  No data found for {symbol}")
                return pd.DataFrame()
            
            # Cache the data
            self.cache[cache_key] = (data, current_time)
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_price_change(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price changes and percentages"""
        if data.empty or len(data) < 2:
            return {'change': 0, 'change_pct': 0, 'current_price': 0}
        
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2]
        change = current_price - previous_price
        change_pct = (change / previous_price) * 100
        
        return {
            'change': change,
            'change_pct': change_pct,
            'current_price': current_price
        }

class MarketInsightEngine:
    """Generates AI-powered market insights by correlating price and sentiment data"""
    
    def __init__(self):
        self.stock_manager = StockDataManager()
        self.news_manager = NewsManager()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Complete analysis of a stock symbol"""
        print(f"🔍 Analyzing {symbol}...")
        
        # Fetch stock data
        stock_data = self.stock_manager.get_stock_data(symbol, period="1d", interval="5m")
        if stock_data.empty:
            return {'error': f'No stock data available for {symbol}'}
        
        # Calculate price movements
        price_info = self.stock_manager.calculate_price_change(stock_data)
        
        # Fetch and analyze news
        news_items = self.news_manager.fetch_rss_news(symbol, limit=15)
        
        # Analyze sentiment of news
        sentiment_scores = []
        for news in news_items[:10]:  # Analyze top 10 news items
            text = f"{news['title']} {news['description']}"
            sentiment = self.sentiment_analyzer.analyze_sentiment(text)
            sentiment_scores.append(sentiment)
        
        # Calculate average sentiment
        if sentiment_scores:
            avg_sentiment = {
                'positive': np.mean([s['positive'] for s in sentiment_scores]),
                'negative': np.mean([s['negative'] for s in sentiment_scores]),
                'neutral': np.mean([s['neutral'] for s in sentiment_scores]),
                'compound': np.mean([s['compound'] for s in sentiment_scores])
            }
        else:
            avg_sentiment = {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0}
        
        # Generate insights
        insights = self._generate_insights(symbol, price_info, avg_sentiment, news_items)
        
        return {
            'symbol': symbol,
            'price_info': price_info,
            'sentiment': avg_sentiment,
            'news_count': len(news_items),
            'insights': insights,
            'timestamp': datetime.now().isoformat(),
            'stock_data': stock_data
        }
    
    def _generate_insights(self, symbol: str, price_info: Dict, sentiment: Dict, news_items: List) -> str:
        """Generate human-readable market insights"""
        change_pct = price_info['change_pct']
        current_price = price_info['current_price']
        sentiment_score = sentiment['compound']
        
        # Determine price movement direction
        if abs(change_pct) < 0.5:
            price_direction = "relatively stable"
        elif change_pct > 0:
            price_direction = f"up {change_pct:.2f}%"
        else:
            price_direction = f"down {abs(change_pct):.2f}%"
        
        # Determine sentiment direction
        if sentiment_score > 0.1:
            sentiment_direction = "positive"
        elif sentiment_score < -0.1:
            sentiment_direction = "negative"
        else:
            sentiment_direction = "neutral"
        
        # Generate correlation insights
        correlation_insight = ""
        if abs(change_pct) > 1:  # Significant price movement
            if (change_pct > 0 and sentiment_score > 0) or (change_pct < 0 and sentiment_score < 0):
                correlation_insight = " The price movement aligns with news sentiment, suggesting news-driven trading."
            elif (change_pct > 0 and sentiment_score < 0) or (change_pct < 0 and sentiment_score > 0):
                correlation_insight = " Interestingly, the price movement contradicts news sentiment, possibly indicating technical trading or contrarian investor behavior."
        
        # Key news themes
        news_themes = self._extract_news_themes(news_items)
        theme_text = f" Key news themes include: {', '.join(news_themes)}." if news_themes else ""
        
        insight = (f"{symbol} is trading at ${current_price:.2f}, {price_direction} recently. "
                  f"News sentiment is {sentiment_direction} based on {len(news_items)} recent articles."
                  f"{correlation_insight}{theme_text}")
        
        return insight
    
    def _extract_news_themes(self, news_items: List, limit: int = 3) -> List[str]:
        """Extract key themes from news headlines"""
        if not news_items:
            return []
        
        # Common financial keywords to look for
        theme_keywords = {
            'earnings': ['earnings', 'profit', 'revenue', 'results'],
            'analyst': ['analyst', 'upgrade', 'downgrade', 'rating', 'target'],
            'guidance': ['guidance', 'outlook', 'forecast', 'expects'],
            'product': ['product', 'launch', 'innovation', 'technology'],
            'acquisition': ['acquisition', 'merger', 'deal', 'partnership'],
            'regulatory': ['fda', 'approval', 'regulation', 'compliance']
        }
        
        themes_found = []
        all_text = ' '.join([item['title'].lower() for item in news_items[:5]])
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                themes_found.append(theme)
        
        return themes_found[:limit]

class MarketDashboard:
    """Creates visualizations and charts for market data"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def create_candlestick_chart(self, symbol: str, stock_data: pd.DataFrame, 
                               sentiment_score: float) -> go.Figure:
        """Create interactive candlestick chart with sentiment overlay"""
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price Action', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(stock_data['Close'], stock_data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add sentiment indicator
        sentiment_color = 'green' if sentiment_score > 0 else 'red' if sentiment_score < 0 else 'gray'
        fig.add_annotation(
            text=f"Sentiment: {sentiment_score:.3f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12, color=sentiment_color),
            bgcolor="white",
            bordercolor=sentiment_color,
            borderwidth=1
        )
        
        fig.update_layout(
            title=f'{symbol} - Market Analysis Dashboard',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_sentiment_overview(self, analyses: List[Dict]) -> go.Figure:
        """Create sentiment overview chart for multiple symbols"""
        symbols = [a['symbol'] for a in analyses]
        sentiments = [a['sentiment']['compound'] for a in analyses]
        changes = [a['price_info']['change_pct'] for a in analyses]
        
        fig = go.Figure()
        
        # Color code based on sentiment
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sentiments]
        
        fig.add_trace(go.Scatter(
            x=sentiments,
            y=changes,
            mode='markers+text',
            text=symbols,
            textposition="middle center",
            marker=dict(
                size=60,
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            name='Stocks'
        ))
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Price Change vs News Sentiment",
            xaxis_title="News Sentiment Score",
            yaxis_title="Price Change (%)",
            width=800,
            height=600
        )
        
        return fig

def run_market_monitor(symbols: List[str] = ['SPY', 'AAPL', 'NVDA', 'TSLA'], 
                      refresh_interval: int = 300):
    """Main function to run the market monitor"""
    
    print("🚀 Financial Market Monitor Starting...")
    print("=" * 50)
    
    insight_engine = MarketInsightEngine()
    dashboard = MarketDashboard()
    
    while True:
        try:
            print(f"\n📊 Market Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 50)
            
            analyses = []
            
            for symbol in symbols:
                analysis = insight_engine.analyze_symbol(symbol)
                if 'error' not in analysis:
                    analyses.append(analysis)
                    
                    # Print insights
                    price_info = analysis['price_info']
                    sentiment = analysis['sentiment']
                    
                    print(f"\n{symbol}: ${price_info['current_price']:.2f} "
                          f"({price_info['change_pct']:+.2f}%)")
                    print(f"Sentiment: {sentiment['compound']:.3f} "
                          f"(Pos: {sentiment['positive']:.2f}, Neg: {sentiment['negative']:.2f})")
                    print(f"📰 {analysis['insights']}")
                    
                    # Create and save individual chart
                    if not analysis['stock_data'].empty:
                        fig = dashboard.create_candlestick_chart(
                            symbol, 
                            analysis['stock_data'], 
                            sentiment['compound']
                        )
                        fig.write_html(f"{symbol}_chart.html")
                        print(f"📈 Chart saved as {symbol}_chart.html")
                
                time.sleep(2)  # Rate limiting
            
            # Create overview chart
            if len(analyses) > 1:
                overview_fig = dashboard.create_sentiment_overview(analyses)
                overview_fig.write_html("market_overview.html")
                print(f"\n📊 Market overview saved as market_overview.html")
            
            print(f"\n⏰ Next update in {refresh_interval//60} minutes...")
            print("=" * 50)
            
            # Sleep until next refresh
            time.sleep(refresh_interval)
            
        except KeyboardInterrupt:
            print("\n\n👋 Market Monitor stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            print("Retrying in 30 seconds...")
            time.sleep(30)

# Streamlit Dashboard (Optional)
def create_streamlit_dashboard():
    """Create a Streamlit web dashboard"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Install with: pip install streamlit")
        return
    
    st.set_page_config(page_title="Financial Market Monitor", layout="wide")
    st.title("🚀 Financial Market Monitor")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    symbols = st.sidebar.multiselect(
        "Select Symbols",
        ['SPY', 'AAPL', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 'MSFT'],
        default=['SPY', 'AAPL', 'NVDA']
    )
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5 min)", value=False)
    
    if st.sidebar.button("Analyze Now") or auto_refresh:
        insight_engine = MarketInsightEngine()
        dashboard = MarketDashboard()
        
        analyses = []
        
        # Progress bar
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            with st.spinner(f"Analyzing {symbol}..."):
                analysis = insight_engine.analyze_symbol(symbol)
                if 'error' not in analysis:
                    analyses.append(analysis)
            
            progress_bar.progress((i + 1) / len(symbols))
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Market Overview")
            if len(analyses) > 1:
                overview_fig = dashboard.create_sentiment_overview(analyses)
                st.plotly_chart(overview_fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Individual Analysis")
            for analysis in analyses:
                with st.expander(f"{analysis['symbol']} Analysis"):
                    price_info = analysis['price_info']
                    sentiment = analysis['sentiment']
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Price", f"${price_info['current_price']:.2f}", 
                               f"{price_info['change_pct']:+.2f}%")
                    col_b.metric("Sentiment", f"{sentiment['compound']:.3f}")
                    col_c.metric("News Count", analysis['news_count'])
                    
                    st.write("**Insights:**", analysis['insights'])
                    
                    # Individual chart
                    if not analysis['stock_data'].empty:
                        fig = dashboard.create_candlestick_chart(
                            analysis['symbol'], 
                            analysis['stock_data'], 
                            sentiment['compound']
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        if auto_refresh:
            time.sleep(300)  # 5 minutes
            st.rerun()

if __name__ == "__main__":
    print("""
🚀 Financial Market Monitor
===========================

Features:
- Real-time stock data (15-20min delay via yfinance)
- Financial news sentiment analysis using FinBERT
- AI-generated market insights
- Interactive candlestick charts
- Price-sentiment correlation analysis

Usage Options:
1. Command Line: python financial_monitor.py
2. Streamlit Dashboard: streamlit run financial_monitor.py
3. Jupyter Notebook: Import and use individual classes

Requirements:
pip install yfinance pandas numpy matplotlib plotly seaborn beautifulsoup4 feedparser transformers torch streamlit

Optional API Keys:
- NewsAPI (free tier): For additional news sources
- Set as environment variable: export NEWSAPI_KEY=your_key_here

""")
    
    # Check if running in Streamlit
    try:
        import streamlit as st
        if hasattr(st, 'runtime'):
            create_streamlit_dashboard()
        else:
            run_market_monitor()
    except:
        run_market_monitor()