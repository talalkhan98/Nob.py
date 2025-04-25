import requests
import pandas as pd
import numpy as np
import nltk
import re
import os
import json
import time
from datetime import datetime, timedelta
import threading
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalysisSystem:
    """
    Advanced sentiment analysis system that analyzes news, social media, and market data
    to predict market movements and enhance trading decisions.
    """
    
    def __init__(self):
        """Initialize the SentimentAnalysisSystem."""
        self.news_sources = []
        self.social_sources = []
        self.sentiment_data = {}
        self.running = False
        self.thread = None
        self.scan_interval = 300  # 5 minutes
        self.sentiment_history = {}
        self.correlation_data = {}
        self.impact_factors = {}
        self.nlp_models = {}
        self.keywords = {}
        self.market_data = {}
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sentiment')
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        
        # Configure logging
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        self.log_file = os.path.join(logs_dir, 'sentiment.log')
        
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('sentiment')
        
        # Initialize NLP components
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP components."""
        try:
            # Download NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize VADER sentiment analyzer
            self.nlp_models['vader'] = SentimentIntensityAnalyzer()
            
            # Initialize TextBlob
            self.nlp_models['textblob'] = TextBlob
            
            # Initialize Hugging Face transformers for financial sentiment
            device = 0 if torch.cuda.is_available() else -1
            
            # Financial sentiment model
            try:
                model_name = "ProsusAI/finbert"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.nlp_models['finbert'] = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
                self.logger.info("Loaded FinBERT model for financial sentiment analysis")
            except Exception as e:
                self.logger.error(f"Error loading FinBERT model: {str(e)}")
                # Fallback to general sentiment model
                try:
                    self.nlp_models['transformer'] = pipeline("sentiment-analysis", device=device)
                    self.logger.info("Loaded general sentiment analysis model as fallback")
                except Exception as e:
                    self.logger.error(f"Error loading fallback sentiment model: {str(e)}")
            
            self.logger.info("Successfully initialized NLP components")
        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {str(e)}")
    
    def configure_news_sources(self, sources):
        """
        Configure news sources for sentiment analysis.
        
        Args:
            sources (list): List of news source configurations
        
        Returns:
            bool: Success status
        """
        try:
            self.news_sources = sources
            self.logger.info(f"Configured {len(sources)} news sources")
            return True
        except Exception as e:
            self.logger.error(f"Error configuring news sources: {str(e)}")
            return False
    
    def configure_social_sources(self, sources):
        """
        Configure social media sources for sentiment analysis.
        
        Args:
            sources (list): List of social media source configurations
        
        Returns:
            bool: Success status
        """
        try:
            self.social_sources = sources
            self.logger.info(f"Configured {len(sources)} social media sources")
            return True
        except Exception as e:
            self.logger.error(f"Error configuring social media sources: {str(e)}")
            return False
    
    def configure_keywords(self, symbol, keywords):
        """
        Configure keywords for a specific symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            keywords (list): List of keywords to track
        
        Returns:
            bool: Success status
        """
        try:
            # Clean symbol for use as key
            clean_symbol = self._clean_symbol(symbol)
            
            # Store keywords
            self.keywords[clean_symbol] = keywords
            
            self.logger.info(f"Configured {len(keywords)} keywords for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error configuring keywords for {symbol}: {str(e)}")
            return False
    
    def _clean_symbol(self, symbol):
        """
        Clean symbol for use as key.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            str: Cleaned symbol
        """
        return symbol.replace('/', '_').replace('-', '_').lower()
    
    def start_sentiment_scanner(self, symbols=None, scan_interval=300):
        """
        Start the sentiment scanner.
        
        Args:
            symbols (list): List of symbols to scan (e.g., ['BTC/USDT', 'ETH/USDT'])
            scan_interval (int): Scan interval in seconds
        
        Returns:
            bool: Success status
        """
        if self.running:
            self.logger.info("Sentiment scanner is already running")
            return False
        
        # If no symbols provided, use common ones
        if not symbols:
            symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'ADA/USDT']
        
        # Configure default keywords if not already configured
        for symbol in symbols:
            clean_symbol = self._clean_symbol(symbol)
            if clean_symbol not in self.keywords:
                base_currency = symbol.split('/')[0]
                self._configure_default_keywords(clean_symbol, base_currency)
        
        self.symbols = symbols
        self.scan_interval = scan_interval
        self.running = True
        
        # Start scanner thread
        self.thread = threading.Thread(target=self._scanner_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.logger.info(f"Started sentiment scanner for {', '.join(symbols)}")
        return True
    
    def _configure_default_keywords(self, clean_symbol, base_currency):
        """
        Configure default keywords for a symbol.
        
        Args:
            clean_symbol (str): Cleaned symbol
            base_currency (str): Base currency (e.g., 'BTC')
        """
        # Default keywords based on currency
        default_keywords = [
            base_currency,
            base_currency.lower(),
            f"#{base_currency}",
            f"${base_currency}"
        ]
        
        # Add common variations
        if base_currency == 'BTC':
            default_keywords.extend(['Bitcoin', 'bitcoin', 'BITCOIN', '#Bitcoin', '$BTC'])
        elif base_currency == 'ETH':
            default_keywords.extend(['Ethereum', 'ethereum', 'ETHEREUM', '#Ethereum', '$ETH'])
        elif base_currency == 'XRP':
            default_keywords.extend(['Ripple', 'ripple', 'RIPPLE', '#Ripple', '$XRP'])
        elif base_currency == 'SOL':
            default_keywords.extend(['Solana', 'solana', 'SOLANA', '#Solana', '$SOL'])
        elif base_currency == 'ADA':
            default_keywords.extend(['Cardano', 'cardano', 'CARDANO', '#Cardano', '$ADA'])
        
        # Store keywords
        self.keywords[clean_symbol] = default_keywords
        
        self.logger.info(f"Configured default keywords for {clean_symbol}: {default_keywords}")
    
    def stop_sentiment_scanner(self):
        """Stop the sentiment scanner."""
        if not self.running:
            self.logger.info("Sentiment scanner is not running")
            return False
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        self.logger.info("Stopped sentiment scanner")
        return True
    
    def _scanner_loop(self):
        """Internal scanner loop."""
        while self.running:
            try:
                # Scan news sources
                self._scan_news_sources()
                
                # Scan social media sources
                self._scan_social_sources()
                
                # Analyze sentiment data
                self._analyze_sentiment_data()
                
                # Update correlation data
                self._update_correlation_data()
                
                # Save sentiment data
                self._save_sentiment_data()
                
                # Sleep until next scan
                time.sleep(self.scan_interval)
            except Exception as e:
                self.logger.error(f"Error in scanner loop: {str(e)}")
                time.sleep(self.scan_interval)
    
    def _scan_news_sources(self):
        """Scan news sources for sentiment data."""
        if not self.news_sources:
            return
        
        self.logger.info("Scanning news sources...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(10, len(self.news_sources))) as executor:
            results = list(executor.map(self._process_news_source, self.news_sources))
        
        # Process results
        for result in results:
            if result:
                source_id, articles = result
                
                # Process each article
                for article in articles:
                    self._process_article(source_id, article)
        
        self.logger.info("Completed news sources scan")
    
    def _process_news_source(self, source):
        """
        Process a news source.
        
        Args:
            source (dict): News source configuration
        
        Returns:
            tuple: (source_id, articles)
        """
        try:
            source_id = source.get('id')
            source_url = source.get('url')
            source_type = source.get('type', 'rss')
            
            if not source_id or not source_url:
                self.logger.error(f"Invalid news source configuration: {source}")
                return None
            
            # Fetch articles based on source type
            if source_type == 'rss':
                articles = self._fetch_rss_feed(source_url)
            elif source_type == 'api':
                api_key = source.get('api_key')
                articles = self._fetch_news_api(source_url, api_key)
            else:
                self.logger.error(f"Unsupported news source type: {source_type}")
                return None
            
            if not articles:
                self.logger.warning(f"No articles found for news source: {source_id}")
                return None
            
            return (source_id, articles)
        except Exception as e:
            self.logger.error(f"Error processing news source {source.get('id')}: {str(e)}")
            return None
    
    def _fetch_rss_feed(self, url):
        """
        Fetch articles from an RSS feed.
        
        Args:
            url (str): RSS feed URL
        
        Returns:
            list: List of articles
        """
        try:
            # In a real implementation, this would use feedparser or similar
            # For now, simulate with a simple request
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching RSS feed {url}: {response.status_code}")
                return []
            
            # Simulate parsing RSS feed
            # In a real implementation, this would parse XML
            articles = []
            
            # Simulate 5 articles
            for i in range(5):
                articles.append({
                    'title': f"Simulated Article {i+1}",
                    'description': f"This is a simulated article {i+1} for testing purposes.",
                    'link': f"{url}/article{i+1}",
                    'published': datetime.now().isoformat(),
                    'source': url
                })
            
            return articles
        except Exception as e:
            self.logger.error(f"Error fetching RSS feed {url}: {str(e)}")
            return []
    
    def _fetch_news_api(self, url, api_key):
        """
        Fetch articles from a news API.
        
        Args:
            url (str): API URL
            api_key (str): API key
        
        Returns:
            list: List of articles
        """
        try:
            # In a real implementation, this would use the actual API
            # For now, simulate with a simple request
            headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                self.logger.error(f"Error fetching news API {url}: {response.status_code}")
                return []
            
            # Simulate parsing API response
            articles = []
            
            # Simulate 5 articles
            for i in range(5):
                articles.append({
                    'title': f"API Article {i+1}",
                    'description': f"This is an API article {i+1} for testing purposes.",
                    'url': f"{url}/article{i+1}",
                    'publishedAt': datetime.now().isoformat(),
                    'source': {'name': 'API Source'}
                })
            
            return articles
        except Exception as e:
            self.logger.error(f"Error fetching news API {url}: {str(e)}")
            return []
    
    def _process_article(self, source_id, article):
        """
        Process an article for sentiment analysis.
        
        Args:
            source_id (str): Source ID
            article (dict): Article data
        """
        try:
            # Extract article text
            title = article.get('title', '')
            description = article.get('description', '')
            
            # Combine title and description for analysis
            text = f"{title}. {description}"
            
            # Check if article contains keywords for any symbol
            for symbol in self.symbols:
                clean_symbol = self._clean_symbol(symbol)
                
                # Skip if no keywords configured
                if clean_symbol not in self.keywords:
                    continue
                
                # Check if article contains any keywords for this symbol
                keywords = self.keywords[clean_symbol]
                if not any(keyword.lower() in text.lower() for keyword in keywords):
                    continue
                
                # Analyze sentiment
                sentiment = self._analyze_text_sentiment(text)
                
                # Store sentiment data
                if clean_symbol not in self.sentiment_data:
                    self.sentiment_data[clean_symbol] = []
                
                self.sentiment_data[clean_symbol].append({
                    'source': source_id,
                    'type': 'news',
                    'title': title,
                    'text': description,
                    'url': article.get('link', article.get('url', '')),
                    'published': article.get('published', article.get('publishedAt', datetime.now().isoformat())),
                    'sentiment': sentiment,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.logger.info(f"Processed article for {symbol}: {title} - Sentiment: {sentiment['compound']:.2f}")
        except Exception as e:
            self.logger.error(f"Error processing article from {source_id}: {str(e)}")
    
    def _scan_social_sources(self):
        """Scan social media sources for sentiment data."""
        if not self.social_sources:
            return
        
        self.logger.info("Scanning social media sources...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(10, len(self.social_sources))) as executor:
            results = list(executor.map(self._process_social_source, self.social_sources))
        
        # Process results
        for result in results:
            if result:
                source_id, posts = result
                
                # Process each post
                for post in posts:
                    self._process_social_post(source_id, post)
        
        self.logger.info("Completed social media sources scan")
    
    def _process_social_source(self, source):
        """
        Process a social media source.
        
        Args:
            source (dict): Social media source configuration
        
        Returns:
            tuple: (source_id, posts)
        """
        try:
            source_id = source.get('id')
            source_url = source.get('url')
            source_type = source.get('type', 'twitter')
            
            if not source_id or not source_url:
                self.logger.error(f"Invalid social media source configuration: {source}")
                return None
            
            # Fetch posts based on source type
            if source_type == 'twitter':
                api_key = source.get('api_key')
                posts = self._fetch_twitter_posts(source_url, api_key)
            elif source_type == 'reddit':
                posts = self._fetch_reddit_posts(source_url)
            else:
                self.logger.error(f"Unsupported social media source type: {source_type}")
                return None
            
            if not posts:
                self.logger.warning(f"No posts found for social media source: {source_id}")
                return None
            
            return (source_id, posts)
        except Exception as e:
            self.logger.error(f"Error processing social media source {source.get('id')}: {str(e)}")
            return None
    
    def _fetch_twitter_posts(self, url, api_key):
        """
        Fetch posts from Twitter.
        
        Args:
            url (str): Twitter API URL
            api_key (str): API key
        
        Returns:
            list: List of posts
        """
        try:
            # In a real implementation, this would use the Twitter API
            # For now, simulate with a simple request
            headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
            
            # Simulate posts for each symbol
            posts = []
            
            for symbol in self.symbols:
                clean_symbol = self._clean_symbol(symbol)
                
                # Skip if no keywords configured
                if clean_symbol not in self.keywords:
                    continue
                
                # Get keywords for this symbol
                keywords = self.keywords[clean_symbol]
                
                # Simulate 3 posts per keyword (limit to first 3 keywords)
                for keyword in keywords[:3]:
                    for i in range(3):
                        sentiment_values = ['positive', 'neutral', 'negative']
                        sentiment_value = sentiment_values[i % 3]
                        
                        posts.append({
                            'id': f"{clean_symbol}_{keyword}_{i}",
                            'text': f"Simulated {sentiment_value} tweet about {keyword} for testing purposes. #{keyword} ${symbol.split('/')[0]}",
                            'created_at': datetime.now().isoformat(),
                            'user': {'screen_name': f"user_{i}", 'followers_count': 1000 * (i + 1)},
                            'retweet_count': 10 * (i + 1),
                            'favorite_count': 20 * (i + 1),
                            'symbol': symbol
                        })
            
            return posts
        except Exception as e:
            self.logger.error(f"Error fetching Twitter posts: {str(e)}")
            return []
    
    def _fetch_reddit_posts(self, url):
        """
        Fetch posts from Reddit.
        
        Args:
            url (str): Reddit API URL
        
        Returns:
            list: List of posts
        """
        try:
            # In a real implementation, this would use the Reddit API
            # For now, simulate with a simple request
            
            # Simulate posts for each symbol
            posts = []
            
            for symbol in self.symbols:
                clean_symbol = self._clean_symbol(symbol)
                
                # Skip if no keywords configured
                if clean_symbol not in self.keywords:
                    continue
                
                # Get keywords for this symbol
                keywords = self.keywords[clean_symbol]
                
                # Simulate 2 posts per keyword (limit to first 3 keywords)
                for keyword in keywords[:3]:
                    for i in range(2):
                        sentiment_values = ['positive', 'neutral', 'negative', 'very positive', 'very negative']
                        sentiment_value = sentiment_values[i % 5]
                        
                        posts.append({
                            'id': f"{clean_symbol}_{keyword}_{i}",
                            'title': f"Simulated {sentiment_value} Reddit post about {keyword}",
                            'selftext': f"This is a simulated {sentiment_value} Reddit post about {keyword} for testing purposes. #{keyword} ${symbol.split('/')[0]}",
                            'created_utc': time.time(),
                            'author': f"redditor_{i}",
                            'score': 50 * (i + 1),
                            'num_comments': 5 * (i + 1),
                            'subreddit': 'cryptocurrency',
                            'symbol': symbol
                        })
            
            return posts
        except Exception as e:
            self.logger.error(f"Error fetching Reddit posts: {str(e)}")
            return []
    
    def _process_social_post(self, source_id, post):
        """
        Process a social media post for sentiment analysis.
        
        Args:
            source_id (str): Source ID
            post (dict): Post data
        """
        try:
            # Extract post text
            if source_id.startswith('twitter'):
                text = post.get('text', '')
                url = f"https://twitter.com/{post.get('user', {}).get('screen_name', 'user')}/status/{post.get('id', '')}"
                published = post.get('created_at', datetime.now().isoformat())
                title = text[:50] + '...' if len(text) > 50 else text
            elif source_id.startswith('reddit'):
                title = post.get('title', '')
                text = post.get('selftext', '')
                url = f"https://reddit.com/r/{post.get('subreddit', 'cryptocurrency')}/comments/{post.get('id', '')}"
                published = datetime.fromtimestamp(post.get('created_utc', time.time())).isoformat()
            else:
                self.logger.error(f"Unsupported social media source: {source_id}")
                return
            
            # Get symbol from post
            symbol = post.get('symbol')
            if not symbol:
                return
            
            clean_symbol = self._clean_symbol(symbol)
            
            # Analyze sentiment
            sentiment = self._analyze_text_sentiment(f"{title}. {text}")
            
            # Store sentiment data
            if clean_symbol not in self.sentiment_data:
                self.sentiment_data[clean_symbol] = []
            
            self.sentiment_data[clean_symbol].append({
                'source': source_id,
                'type': 'social',
                'title': title,
                'text': text,
                'url': url,
                'published': published,
                'sentiment': sentiment,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Processed social post for {symbol}: {title[:30]}... - Sentiment: {sentiment['compound']:.2f}")
        except Exception as e:
            self.logger.error(f"Error processing social post from {source_id}: {str(e)}")
    
    def _analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            dict: Sentiment scores
        """
        try:
            # Use VADER for general sentiment
            vader_sentiment = self.nlp_models['vader'].polarity_scores(text)
            
            # Use TextBlob for additional sentiment
            textblob = self.nlp_models['textblob'](text)
            textblob_polarity = textblob.sentiment.polarity
            textblob_subjectivity = textblob.sentiment.subjectivity
            
            # Use FinBERT for financial sentiment if available
            finbert_sentiment = None
            if 'finbert' in self.nlp_models:
                try:
                    # Truncate text if too long
                    truncated_text = text[:512] if len(text) > 512 else text
                    finbert_result = self.nlp_models['finbert'](truncated_text)[0]
                    finbert_sentiment = {
                        'label': finbert_result['label'],
                        'score': finbert_result['score']
                    }
                except Exception as e:
                    self.logger.error(f"Error using FinBERT: {str(e)}")
            
            # Combine sentiment scores
            sentiment = {
                'vader': vader_sentiment,
                'textblob': {
                    'polarity': textblob_polarity,
                    'subjectivity': textblob_subjectivity
                },
                'compound': vader_sentiment['compound'],  # Use VADER compound as primary score
                'finbert': finbert_sentiment
            }
            
            return sentiment
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'vader': {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0},
                'textblob': {'polarity': 0, 'subjectivity': 0},
                'compound': 0,
                'finbert': None
            }
    
    def _analyze_sentiment_data(self):
        """Analyze sentiment data for all symbols."""
        for symbol in self.symbols:
            clean_symbol = self._clean_symbol(symbol)
            
            # Skip if no sentiment data
            if clean_symbol not in self.sentiment_data or not self.sentiment_data[clean_symbol]:
                continue
            
            # Get sentiment data for this symbol
            sentiment_data = self.sentiment_data[clean_symbol]
            
            # Calculate aggregate sentiment
            compound_scores = [item['sentiment']['compound'] for item in sentiment_data]
            avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
            
            # Calculate weighted sentiment based on source type and recency
            weighted_scores = []
            weights = []
            
            for item in sentiment_data:
                # Base weight
                weight = 1.0
                
                # Adjust weight based on source type
                if item['type'] == 'news':
                    weight *= 1.5  # News has higher weight
                
                # Adjust weight based on recency
                published = datetime.fromisoformat(item['published']) if isinstance(item['published'], str) else item['published']
                age_hours = (datetime.now() - published).total_seconds() / 3600
                recency_factor = max(0.1, min(1.0, 1.0 - (age_hours / 24)))  # Decay over 24 hours
                weight *= recency_factor
                
                weighted_scores.append(item['sentiment']['compound'] * weight)
                weights.append(weight)
            
            weighted_avg = sum(weighted_scores) / sum(weights) if weights else 0
            
            # Store in sentiment history
            if clean_symbol not in self.sentiment_history:
                self.sentiment_history[clean_symbol] = []
            
            self.sentiment_history[clean_symbol].append({
                'timestamp': datetime.now().isoformat(),
                'avg_compound': avg_compound,
                'weighted_avg': weighted_avg,
                'data_points': len(sentiment_data),
                'news_count': sum(1 for item in sentiment_data if item['type'] == 'news'),
                'social_count': sum(1 for item in sentiment_data if item['type'] == 'social')
            })
            
            # Keep only last 7 days of history
            cutoff = datetime.now() - timedelta(days=7)
            self.sentiment_history[clean_symbol] = [
                item for item in self.sentiment_history[clean_symbol]
                if datetime.fromisoformat(item['timestamp']) > cutoff
            ]
            
            self.logger.info(f"Analyzed sentiment for {symbol}: Avg={avg_compound:.2f}, Weighted={weighted_avg:.2f}, Points={len(sentiment_data)}")
    
    def _update_correlation_data(self):
        """Update correlation between sentiment and price movements."""
        # Skip if no market data
        if not self.market_data:
            return
        
        for symbol in self.symbols:
            clean_symbol = self._clean_symbol(symbol)
            
            # Skip if no sentiment history or market data
            if (clean_symbol not in self.sentiment_history or 
                not self.sentiment_history[clean_symbol] or
                clean_symbol not in self.market_data or
                not self.market_data[clean_symbol]):
                continue
            
            # Get sentiment history and market data
            sentiment_history = self.sentiment_history[clean_symbol]
            market_data = self.market_data[clean_symbol]
            
            # Need at least 5 data points
            if len(sentiment_history) < 5 or len(market_data) < 5:
                continue
            
            try:
                # Create DataFrames
                sentiment_df = pd.DataFrame(sentiment_history)
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                sentiment_df.set_index('timestamp', inplace=True)
                
                market_df = pd.DataFrame(market_data)
                market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
                market_df.set_index('timestamp', inplace=True)
                
                # Resample to hourly data
                sentiment_hourly = sentiment_df.resample('1H').mean().dropna()
                market_hourly = market_df.resample('1H').mean().dropna()
                
                # Align data
                aligned_data = pd.merge(
                    sentiment_hourly[['weighted_avg']], 
                    market_hourly[['price_change_pct']], 
                    left_index=True, 
                    right_index=True, 
                    how='inner'
                )
                
                # Calculate correlation
                if len(aligned_data) >= 5:
                    correlation = aligned_data['weighted_avg'].corr(aligned_data['price_change_pct'])
                    
                    # Calculate lagged correlations (sentiment leading price by 1-6 hours)
                    lagged_correlations = []
                    for lag in range(1, 7):
                        lagged_data = pd.merge(
                            sentiment_hourly[['weighted_avg']].shift(lag), 
                            market_hourly[['price_change_pct']], 
                            left_index=True, 
                            right_index=True, 
                            how='inner'
                        )
                        
                        if len(lagged_data) >= 5:
                            lagged_corr = lagged_data['weighted_avg'].corr(lagged_data['price_change_pct'])
                            lagged_correlations.append((lag, lagged_corr))
                    
                    # Find best lag
                    best_lag = 0
                    best_corr = correlation
                    
                    for lag, corr in lagged_correlations:
                        if abs(corr) > abs(best_corr):
                            best_lag = lag
                            best_corr = corr
                    
                    # Store correlation data
                    self.correlation_data[clean_symbol] = {
                        'timestamp': datetime.now().isoformat(),
                        'correlation': correlation,
                        'best_lag': best_lag,
                        'best_correlation': best_corr,
                        'data_points': len(aligned_data)
                    }
                    
                    # Calculate impact factor (how much sentiment affects price)
                    if abs(best_corr) > 0.3:  # Only if correlation is significant
                        # Simple linear regression
                        X = aligned_data['weighted_avg'].values.reshape(-1, 1)
                        y = aligned_data['price_change_pct'].values
                        
                        # Add constant for intercept
                        X = np.hstack((np.ones((X.shape[0], 1)), X))
                        
                        # Solve for coefficients
                        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                        
                        # Store impact factor
                        self.impact_factors[clean_symbol] = {
                            'intercept': coeffs[0],
                            'slope': coeffs[1],
                            'correlation': best_corr
                        }
                    
                    self.logger.info(f"Updated correlation for {symbol}: Corr={correlation:.2f}, Best Lag={best_lag}h, Best Corr={best_corr:.2f}")
            except Exception as e:
                self.logger.error(f"Error updating correlation for {symbol}: {str(e)}")
    
    def _save_sentiment_data(self):
        """Save sentiment data to disk."""
        try:
            # Save sentiment history
            history_file = os.path.join(self.data_dir, 'sentiment_history.json')
            with open(history_file, 'w') as f:
                json.dump(self.sentiment_history, f)
            
            # Save correlation data
            correlation_file = os.path.join(self.data_dir, 'correlation_data.json')
            with open(correlation_file, 'w') as f:
                json.dump(self.correlation_data, f)
            
            # Save impact factors
            impact_file = os.path.join(self.data_dir, 'impact_factors.json')
            with open(impact_file, 'w') as f:
                json.dump(self.impact_factors, f)
            
            self.logger.info("Saved sentiment data to disk")
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {str(e)}")
    
    def load_sentiment_data(self):
        """
        Load sentiment data from disk.
        
        Returns:
            bool: Success status
        """
        try:
            # Load sentiment history
            history_file = os.path.join(self.data_dir, 'sentiment_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.sentiment_history = json.load(f)
            
            # Load correlation data
            correlation_file = os.path.join(self.data_dir, 'correlation_data.json')
            if os.path.exists(correlation_file):
                with open(correlation_file, 'r') as f:
                    self.correlation_data = json.load(f)
            
            # Load impact factors
            impact_file = os.path.join(self.data_dir, 'impact_factors.json')
            if os.path.exists(impact_file):
                with open(impact_file, 'r') as f:
                    self.impact_factors = json.load(f)
            
            self.logger.info("Loaded sentiment data from disk")
            return True
        except Exception as e:
            self.logger.error(f"Error loading sentiment data: {str(e)}")
            return False
    
    def update_market_data(self, symbol, market_data):
        """
        Update market data for correlation analysis.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            market_data (list): List of market data points
        
        Returns:
            bool: Success status
        """
        try:
            clean_symbol = self._clean_symbol(symbol)
            
            # Store market data
            self.market_data[clean_symbol] = market_data
            
            self.logger.info(f"Updated market data for {symbol}: {len(market_data)} points")
            return True
        except Exception as e:
            self.logger.error(f"Error updating market data for {symbol}: {str(e)}")
            return False
    
    def get_sentiment_score(self, symbol):
        """
        Get current sentiment score for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            dict: Sentiment score data
        """
        try:
            clean_symbol = self._clean_symbol(symbol)
            
            # Check if sentiment history exists
            if clean_symbol not in self.sentiment_history or not self.sentiment_history[clean_symbol]:
                return {
                    'symbol': symbol,
                    'score': 0,
                    'weighted_score': 0,
                    'data_points': 0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get latest sentiment
            latest = self.sentiment_history[clean_symbol][-1]
            
            # Get correlation data
            correlation = self.correlation_data.get(clean_symbol, {})
            
            # Get impact factor
            impact = self.impact_factors.get(clean_symbol, {})
            
            return {
                'symbol': symbol,
                'score': latest['avg_compound'],
                'weighted_score': latest['weighted_avg'],
                'data_points': latest['data_points'],
                'news_count': latest['news_count'],
                'social_count': latest['social_count'],
                'timestamp': latest['timestamp'],
                'correlation': correlation.get('correlation', 0),
                'best_lag': correlation.get('best_lag', 0),
                'best_correlation': correlation.get('best_correlation', 0),
                'impact_factor': impact.get('slope', 0) if impact else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting sentiment score for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'score': 0,
                'weighted_score': 0,
                'data_points': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_sentiment_history(self, symbol, days=7):
        """
        Get sentiment history for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            days (int): Number of days of history
        
        Returns:
            list: Sentiment history
        """
        try:
            clean_symbol = self._clean_symbol(symbol)
            
            # Check if sentiment history exists
            if clean_symbol not in self.sentiment_history or not self.sentiment_history[clean_symbol]:
                return []
            
            # Filter by date
            cutoff = datetime.now() - timedelta(days=days)
            history = [
                item for item in self.sentiment_history[clean_symbol]
                if datetime.fromisoformat(item['timestamp']) > cutoff
            ]
            
            return history
        except Exception as e:
            self.logger.error(f"Error getting sentiment history for {symbol}: {str(e)}")
            return []
    
    def get_sentiment_impact(self, symbol, sentiment_score=None):
        """
        Get predicted price impact based on sentiment.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            sentiment_score (float): Optional sentiment score to use
        
        Returns:
            dict: Predicted impact
        """
        try:
            clean_symbol = self._clean_symbol(symbol)
            
            # Get impact factor
            if clean_symbol not in self.impact_factors:
                return {
                    'symbol': symbol,
                    'predicted_impact': 0,
                    'confidence': 0
                }
            
            impact = self.impact_factors[clean_symbol]
            
            # Get sentiment score if not provided
            if sentiment_score is None:
                sentiment_data = self.get_sentiment_score(symbol)
                sentiment_score = sentiment_data['weighted_score']
            
            # Calculate predicted impact
            predicted_impact = impact['intercept'] + impact['slope'] * sentiment_score
            
            # Calculate confidence based on correlation strength
            confidence = min(100, abs(impact['correlation']) * 100)
            
            return {
                'symbol': symbol,
                'predicted_impact': predicted_impact,
                'confidence': confidence,
                'sentiment_score': sentiment_score,
                'correlation': impact['correlation'],
                'impact_factor': impact['slope']
            }
        except Exception as e:
            self.logger.error(f"Error getting sentiment impact for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'predicted_impact': 0,
                'confidence': 0,
                'error': str(e)
            }
    
    def generate_sentiment_report(self, symbol):
        """
        Generate a detailed sentiment report for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
        
        Returns:
            dict: Sentiment report
        """
        try:
            clean_symbol = self._clean_symbol(symbol)
            
            # Get sentiment data
            sentiment_score = self.get_sentiment_score(symbol)
            sentiment_history = self.get_sentiment_history(symbol)
            sentiment_impact = self.get_sentiment_impact(symbol, sentiment_score['weighted_score'])
            
            # Get recent sentiment data
            if clean_symbol in self.sentiment_data and self.sentiment_data[clean_symbol]:
                recent_data = sorted(
                    self.sentiment_data[clean_symbol],
                    key=lambda x: x['timestamp'],
                    reverse=True
                )[:10]  # Get 10 most recent items
            else:
                recent_data = []
            
            # Generate report
            report = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_sentiment': sentiment_score,
                'predicted_impact': sentiment_impact,
                'sentiment_trend': self._analyze_sentiment_trend(sentiment_history),
                'recent_data': recent_data,
                'news_sentiment': self._analyze_source_sentiment(recent_data, 'news'),
                'social_sentiment': self._analyze_source_sentiment(recent_data, 'social'),
                'keywords': self.keywords.get(clean_symbol, [])
            }
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating sentiment report for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _analyze_sentiment_trend(self, history):
        """
        Analyze sentiment trend from history.
        
        Args:
            history (list): Sentiment history
        
        Returns:
            dict: Trend analysis
        """
        if not history or len(history) < 2:
            return {
                'direction': 'neutral',
                'strength': 0,
                'change_24h': 0
            }
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x['timestamp'])
        
        # Get values
        values = [item['weighted_avg'] for item in sorted_history]
        
        # Calculate trend
        if len(values) >= 3:
            # Simple linear regression
            x = np.arange(len(values))
            y = np.array(values)
            
            # Add constant for intercept
            X = np.vstack((np.ones(len(x)), x)).T
            
            # Solve for coefficients
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            
            slope = coeffs[1]
        else:
            slope = values[-1] - values[0]
        
        # Determine direction
        if slope > 0.01:
            direction = 'bullish'
        elif slope < -0.01:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Calculate strength (0-100)
        strength = min(100, abs(slope) * 100)
        
        # Calculate 24h change
        last_24h = [
            item for item in sorted_history
            if datetime.fromisoformat(item['timestamp']) > datetime.now() - timedelta(hours=24)
        ]
        
        if last_24h and len(last_24h) >= 2:
            change_24h = last_24h[-1]['weighted_avg'] - last_24h[0]['weighted_avg']
        else:
            change_24h = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'change_24h': change_24h
        }
    
    def _analyze_source_sentiment(self, data, source_type):
        """
        Analyze sentiment by source type.
        
        Args:
            data (list): Sentiment data
            source_type (str): Source type ('news' or 'social')
        
        Returns:
            dict: Source sentiment analysis
        """
        # Filter by source type
        filtered_data = [item for item in data if item['type'] == source_type]
        
        if not filtered_data:
            return {
                'count': 0,
                'avg_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Calculate metrics
        sentiments = [item['sentiment']['compound'] for item in filtered_data]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Count by sentiment category
        positive_count = sum(1 for s in sentiments if s > 0.05)
        negative_count = sum(1 for s in sentiments if s < -0.05)
        neutral_count = sum(1 for s in sentiments if -0.05 <= s <= 0.05)
        
        return {
            'count': len(filtered_data),
            'avg_sentiment': avg_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_pct': positive_count / len(filtered_data) * 100 if filtered_data else 0,
            'negative_pct': negative_count / len(filtered_data) * 100 if filtered_data else 0,
            'neutral_pct': neutral_count / len(filtered_data) * 100 if filtered_data else 0
        }
    
    def plot_sentiment_history(self, symbol, days=7):
        """
        Plot sentiment history for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            days (int): Number of days of history
        
        Returns:
            str: Path to saved plot
        """
        try:
            # Get sentiment history
            history = self.get_sentiment_history(symbol, days)
            
            if not history:
                self.logger.error(f"No sentiment history for {symbol}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot sentiment
            plt.plot(df.index, df['weighted_avg'], 'b-', label='Weighted Sentiment')
            plt.plot(df.index, df['avg_compound'], 'g--', label='Average Sentiment')
            
            # Add zero line
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Add shaded regions for sentiment categories
            plt.axhspan(0.05, 1, alpha=0.2, color='green', label='Positive')
            plt.axhspan(-0.05, 0.05, alpha=0.2, color='gray', label='Neutral')
            plt.axhspan(-1, -0.05, alpha=0.2, color='red', label='Negative')
            
            # Add data points count as area
            plt.fill_between(df.index, 0, df['data_points'] / df['data_points'].max(), 
                            alpha=0.2, color='purple', label='Data Points (scaled)')
            
            # Add labels and title
            plt.title(f'Sentiment Analysis for {symbol} - Last {days} Days')
            plt.xlabel('Date')
            plt.ylabel('Sentiment Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_file = os.path.join(plot_dir, f'{clean_symbol}_sentiment_{days}d.png')
            plt.savefig(plot_file)
            plt.close()
            
            return plot_file
        except Exception as e:
            self.logger.error(f"Error plotting sentiment history for {symbol}: {str(e)}")
            return None
