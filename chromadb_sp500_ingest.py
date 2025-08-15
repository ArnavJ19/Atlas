# chromadb_sp500_ingest.py
"""
S&P 500 Data Ingestion using ChromaDB for vector storage
Utilizes ALL fields from yf.Ticker.info with ChromaDB's advanced features
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import hashlib

import pandas as pd
import numpy as np
import yfinance as yf
import networkx as nx
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STORE_DIR = Path("store")
STORE_DIR.mkdir(exist_ok=True)
CHROMA_DIR = STORE_DIR / "chromadb"

# =====================================================
# CHROMADB SETUP
# =====================================================

class ChromaDBManager:
    """Manages ChromaDB operations for S&P 500 data"""
    
    def __init__(self, persist_directory: str = str(CHROMA_DIR)):
        """Initialize ChromaDB with persistence"""
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Use sentence-transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create collections for different data types
        self.collections = {}
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize or get existing collections"""
        # Main company collection
        self.collections['companies'] = self.client.get_or_create_collection(
            name="sp500_companies",
            embedding_function=self.embedding_function,
            metadata={"description": "S&P 500 company profiles and metrics"}
        )
        
        # Financial metrics collection
        self.collections['financials'] = self.client.get_or_create_collection(
            name="sp500_financials",
            embedding_function=self.embedding_function,
            metadata={"description": "Detailed financial metrics and ratios"}
        )
        
        # Trading data collection
        self.collections['trading'] = self.client.get_or_create_collection(
            name="sp500_trading",
            embedding_function=self.embedding_function,
            metadata={"description": "Trading data and price metrics"}
        )
        
        # Sector/Industry collection
        self.collections['sectors'] = self.client.get_or_create_collection(
            name="sp500_sectors",
            embedding_function=self.embedding_function,
            metadata={"description": "Sector and industry analysis"}
        )
        
        logger.info(f"Initialized {len(self.collections)} ChromaDB collections")
    
    def reset_collections(self):
        """Reset all collections (useful for fresh ingestion)"""
        for name in list(self.collections.keys()):
            try:
                self.client.delete_collection(name=self.collections[name].name)
                logger.info(f"Deleted collection: {name}")
            except:
                pass
        
        self._initialize_collections()
        logger.info("All collections reset")
    
    def add_company_data(self, stocks: List[Dict]):
        """Add company data to ChromaDB collections"""
        companies_docs = []
        companies_metadata = []
        companies_ids = []
        
        financials_docs = []
        financials_metadata = []
        financials_ids = []
        
        trading_docs = []
        trading_metadata = []
        trading_ids = []
        
        for stock in stocks:
            if stock['fetch_status'] != 'success':
                continue
            
            ticker = stock['ticker']
            data = stock['organized_data']
            sp500_data = stock['sp500_data']
            scores = stock.get('scores', {})
            
            # Helper functions for safe formatting
            def safe_value(value, default='N/A'):
                return value if value is not None else default
            
            def safe_number(value, decimals=2, default=0):
                if value is None:
                    return default
                try:
                    return round(float(value), decimals)
                except:
                    return default
            
            # 1. Company Overview Document
            company_doc = f"""
{ticker} - {safe_value(data['company_details'].get('longName'), sp500_data['company_name'])}
Sector: {sp500_data['sector']} | Industry: {sp500_data['industry']}
Website: {safe_value(data['company_details'].get('website'))}
Employees: {safe_value(data['company_details'].get('fullTimeEmployees'))}

Business Summary:
{safe_value(data['company_details'].get('longBusinessSummary'), 'No description available')}

Location: {safe_value(data['location'].get('city'))}, {safe_value(data['location'].get('state'))}, {safe_value(data['location'].get('country'))}

Key Metrics:
- Market Cap: ${safe_number(data['valuation'].get('marketCap'), 0)}
- P/E Ratio: {safe_number(data['valuation'].get('trailingPE'))}
- Forward P/E: {safe_number(data['valuation'].get('forwardPE'))}
- Dividend Yield: {safe_number(data['dividends'].get('dividendYield'), 4)}
- Beta: {safe_number(data['risk'].get('beta'))}
- 52-Week Change: {safe_number(data['yearly_metrics'].get('52WeekChange'), 4)}

Investment Score: {safe_number(scores.get('composite_score'))}/100
Percentile Rank: {safe_number(scores.get('percentile_overall'))}%
"""
            
            companies_docs.append(company_doc)
            companies_metadata.append({
                'ticker': ticker,
                'company_name': safe_value(data['company_details'].get('longName'), sp500_data['company_name']),
                'sector': sp500_data['sector'],
                'industry': sp500_data['industry'],
                'market_cap': safe_number(data['valuation'].get('marketCap'), 0),
                'pe_ratio': safe_number(data['valuation'].get('trailingPE')),
                'dividend_yield': safe_number(data['dividends'].get('dividendYield'), 4),
                'composite_score': safe_number(scores.get('composite_score')),
                'percentile_overall': safe_number(scores.get('percentile_overall')),
                'country': safe_value(data['location'].get('country')),
                'exchange': safe_value(data['trading'].get('exchange')),
                'document_type': 'company_overview',
                'last_updated': stock['fetch_timestamp']
            })
            companies_ids.append(f"company_{ticker}")
            
            # 2. Financial Metrics Document
            financial_doc = f"""
Financial Analysis for {ticker}

Valuation Metrics:
- Market Cap: ${safe_number(data['valuation'].get('marketCap'), 0)}
- Enterprise Value: ${safe_number(data['valuation'].get('enterpriseValue'), 0)}
- P/E Ratio: {safe_number(data['valuation'].get('trailingPE'))}
- Forward P/E: {safe_number(data['valuation'].get('forwardPE'))}
- PEG Ratio: {safe_number(data['valuation'].get('trailingPegRatio'))}
- Price/Sales: {safe_number(data['valuation'].get('priceToSalesTrailing12Months'))}
- Price/Book: {safe_number(data['valuation'].get('priceToBook'))}
- EV/EBITDA: {safe_number(data['valuation'].get('enterpriseToEbitda'))}

Profitability:
- Gross Margin: {safe_number(data['financials'].get('grossMargins'), 4)}
- Operating Margin: {safe_number(data['financials'].get('operatingMargins'), 4)}
- Profit Margin: {safe_number(data['financials'].get('profitMargins'), 4)}
- ROE: {safe_number(data['returns'].get('returnOnEquity'), 4)}
- ROA: {safe_number(data['returns'].get('returnOnAssets'), 4)}

Income Statement:
- Revenue: ${safe_number(data['financials'].get('totalRevenue'), 0)}
- Revenue Growth: {safe_number(data['financials'].get('revenueGrowth'), 4)}
- EBITDA: ${safe_number(data['financials'].get('ebitda'), 0)}
- Net Income: ${safe_number(data['financials'].get('netIncomeToCommon'), 0)}
- EPS: ${safe_number(data['earnings'].get('trailingEps'))}

Balance Sheet:
- Total Cash: ${safe_number(data['balance_sheet'].get('totalCash'), 0)}
- Total Debt: ${safe_number(data['balance_sheet'].get('totalDebt'), 0)}
- Debt/Equity: {safe_number(data['balance_sheet'].get('debtToEquity'))}
- Current Ratio: {safe_number(data['balance_sheet'].get('currentRatio'))}
- Quick Ratio: {safe_number(data['balance_sheet'].get('quickRatio'))}

Cash Flow:
- Operating Cash Flow: ${safe_number(data['cash_flow'].get('operatingCashflow'), 0)}
- Free Cash Flow: ${safe_number(data['cash_flow'].get('freeCashflow'), 0)}

Investment Scores:
- Value Score: {safe_number(scores.get('value_score'))}
- Quality Score: {safe_number(scores.get('quality_score'))}
- Growth Score: {safe_number(scores.get('growth_score'))}
- Financial Health Score: {safe_number(scores.get('financial_score'))}
- Momentum Score: {safe_number(scores.get('momentum_score'))}
"""
            
            financials_docs.append(financial_doc)
            financials_metadata.append({
                'ticker': ticker,
                'company_name': safe_value(data['company_details'].get('longName'), sp500_data['company_name']),
                'sector': sp500_data['sector'],
                'pe_ratio': safe_number(data['valuation'].get('trailingPE')),
                'profit_margin': safe_number(data['financials'].get('profitMargins'), 4),
                'roe': safe_number(data['returns'].get('returnOnEquity'), 4),
                'revenue_growth': safe_number(data['financials'].get('revenueGrowth'), 4),
                'debt_to_equity': safe_number(data['balance_sheet'].get('debtToEquity')),
                'free_cashflow': safe_number(data['cash_flow'].get('freeCashflow'), 0),
                'value_score': safe_number(scores.get('value_score')),
                'quality_score': safe_number(scores.get('quality_score')),
                'growth_score': safe_number(scores.get('growth_score')),
                'document_type': 'financial_metrics',
                'last_updated': stock['fetch_timestamp']
            })
            financials_ids.append(f"financials_{ticker}")
            
            # 3. Trading Document
            trading_doc = f"""
Trading Information for {ticker}

Current Trading:
- Current Price: ${safe_number(data['price_metrics'].get('currentPrice'))}
- Previous Close: ${safe_number(data['price_metrics'].get('previousClose'))}
- Day Range: ${safe_number(data['price_metrics'].get('dayLow'))} - ${safe_number(data['price_metrics'].get('dayHigh'))}
- 52 Week Range: ${safe_number(data['yearly_metrics'].get('fiftyTwoWeekLow'))} - ${safe_number(data['yearly_metrics'].get('fiftyTwoWeekHigh'))}
- Volume: {safe_number(data['volume_metrics'].get('volume'), 0)}
- Avg Volume: {safe_number(data['volume_metrics'].get('averageVolume'), 0)}

Moving Averages:
- 50 Day Average: ${safe_number(data['moving_averages'].get('fiftyDayAverage'))}
- 200 Day Average: ${safe_number(data['moving_averages'].get('twoHundredDayAverage'))}
- 50 Day Change: {safe_number(data['moving_averages'].get('fiftyDayAverageChangePercent'), 4)}
- 200 Day Change: {safe_number(data['moving_averages'].get('twoHundredDayAverageChangePercent'), 4)}

Performance:
- 52 Week Change: {safe_number(data['yearly_metrics'].get('52WeekChange'), 4)}
- S&P 52 Week Change: {safe_number(data['yearly_metrics'].get('SandP52WeekChange'), 4)}

Analyst Recommendations:
- Recommendation: {safe_value(data['analyst'].get('recommendationKey'))}
- Target Mean: ${safe_number(data['analyst'].get('targetMeanPrice'))}
- Number of Analysts: {safe_number(data['analyst'].get('numberOfAnalystOpinions'), 0)}

Shares:
- Shares Outstanding: {safe_number(data['shares'].get('sharesOutstanding'), 0)}
- Float: {safe_number(data['shares'].get('floatShares'), 0)}
- Short % of Float: {safe_number(data['shares'].get('shortPercentOfFloat'), 4)}
- Institutional Ownership: {safe_number(data['ownership'].get('heldPercentInstitutions'), 4)}
"""
            
            trading_docs.append(trading_doc)
            trading_metadata.append({
                'ticker': ticker,
                'company_name': safe_value(data['company_details'].get('longName'), sp500_data['company_name']),
                'sector': sp500_data['sector'],
                'current_price': safe_number(data['price_metrics'].get('currentPrice')),
                'volume': safe_number(data['volume_metrics'].get('volume'), 0),
                'fifty_two_week_change': safe_number(data['yearly_metrics'].get('52WeekChange'), 4),
                'fifty_day_avg_change': safe_number(data['moving_averages'].get('fiftyDayAverageChangePercent'), 4),
                'two_hundred_day_avg_change': safe_number(data['moving_averages'].get('twoHundredDayAverageChangePercent'), 4),
                'recommendation': safe_value(data['analyst'].get('recommendationKey')),
                'target_price': safe_number(data['analyst'].get('targetMeanPrice')),
                'institutional_ownership': safe_number(data['ownership'].get('heldPercentInstitutions'), 4),
                'document_type': 'trading_data',
                'last_updated': stock['fetch_timestamp']
            })
            trading_ids.append(f"trading_{ticker}")
        
        # Add to collections in batches
        if companies_docs:
            self.collections['companies'].add(
                documents=companies_docs,
                metadatas=companies_metadata,
                ids=companies_ids
            )
            logger.info(f"Added {len(companies_docs)} company documents to ChromaDB")
        
        if financials_docs:
            self.collections['financials'].add(
                documents=financials_docs,
                metadatas=financials_metadata,
                ids=financials_ids
            )
            logger.info(f"Added {len(financials_docs)} financial documents to ChromaDB")
        
        if trading_docs:
            self.collections['trading'].add(
                documents=trading_docs,
                metadatas=trading_metadata,
                ids=trading_ids
            )
            logger.info(f"Added {len(trading_docs)} trading documents to ChromaDB")
    
    def add_sector_summaries(self, stocks: List[Dict]):
        """Create and add sector summary documents"""
        sector_data = {}
        
        for stock in stocks:
            if stock['fetch_status'] != 'success':
                continue
            
            sector = stock['sp500_data']['sector']
            if sector not in sector_data:
                sector_data[sector] = {
                    'companies': [],
                    'total_market_cap': 0,
                    'scores': [],
                    'pe_ratios': [],
                    'profit_margins': [],
                    'revenue_growth': [],
                    'dividend_yields': []
                }
            
            data = stock['organized_data']
            scores = stock.get('scores', {})
            
            sector_data[sector]['companies'].append(stock['ticker'])
            
            market_cap = data['valuation'].get('marketCap', 0)
            if market_cap:
                sector_data[sector]['total_market_cap'] += market_cap
            
            if scores.get('composite_score'):
                sector_data[sector]['scores'].append(scores['composite_score'])
            
            if data['valuation'].get('trailingPE'):
                sector_data[sector]['pe_ratios'].append(data['valuation']['trailingPE'])
            
            if data['financials'].get('profitMargins'):
                sector_data[sector]['profit_margins'].append(data['financials']['profitMargins'])
            
            if data['financials'].get('revenueGrowth'):
                sector_data[sector]['revenue_growth'].append(data['financials']['revenueGrowth'])
            
            if data['dividends'].get('dividendYield'):
                sector_data[sector]['dividend_yields'].append(data['dividends']['dividendYield'])
        
        # Create sector documents
        sector_docs = []
        sector_metadata = []
        sector_ids = []
        
        for sector, data in sector_data.items():
            avg_score = np.mean(data['scores']) if data['scores'] else 0
            avg_pe = np.mean(data['pe_ratios']) if data['pe_ratios'] else 0
            avg_margin = np.mean(data['profit_margins']) if data['profit_margins'] else 0
            avg_growth = np.mean(data['revenue_growth']) if data['revenue_growth'] else 0
            avg_yield = np.mean(data['dividend_yields']) if data['dividend_yields'] else 0
            
            doc = f"""
Sector Analysis: {sector}

Overview:
- Number of Companies: {len(data['companies'])}
- Total Market Cap: ${data['total_market_cap']/1e9:.2f}B
- Average Investment Score: {avg_score:.2f}/100

Key Metrics (Sector Averages):
- P/E Ratio: {avg_pe:.2f}
- Profit Margin: {avg_margin:.2%}
- Revenue Growth: {avg_growth:.2%}
- Dividend Yield: {avg_yield:.2%}

Top Companies: {', '.join(data['companies'][:10])}

Sector Characteristics:
This sector represents {len(data['companies'])/len(stocks)*100:.1f}% of the S&P 500 companies
and {data['total_market_cap']/sum(s['total_market_cap'] for s in sector_data.values())*100:.1f}% of total market capitalization.
"""
            
            sector_docs.append(doc)
            sector_metadata.append({
                'sector': sector,
                'company_count': len(data['companies']),
                'total_market_cap': data['total_market_cap'],
                'avg_score': avg_score,
                'avg_pe_ratio': avg_pe,
                'avg_profit_margin': avg_margin,
                'avg_revenue_growth': avg_growth,
                'avg_dividend_yield': avg_yield,
                'document_type': 'sector_summary'
            })
            sector_ids.append(f"sector_{sector.replace(' ', '_').lower()}")
        
        if sector_docs:
            self.collections['sectors'].add(
                documents=sector_docs,
                metadatas=sector_metadata,
                ids=sector_ids
            )
            logger.info(f"Added {len(sector_docs)} sector documents to ChromaDB")
    
    def query(self, query_text: str, collection_name: str = 'companies', 
              n_results: int = 5, where: Dict = None):
        """Query ChromaDB collections"""
        collection = self.collections.get(collection_name)
        if not collection:
            logger.error(f"Collection {collection_name} not found")
            return None
        
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_statistics(self):
        """Get statistics about stored data"""
        stats = {}
        for name, collection in self.collections.items():
            count = collection.count()
            stats[name] = {
                'count': count,
                'name': collection.name,
                'metadata': collection.metadata
            }
        return stats


# Keep the existing fetch functions from the original script
# (Replace this comment with the actual functions below)

# =====================================================
# STEP 1: Fetch S&P 500 List
# =====================================================

def fetch_sp500_list() -> pd.DataFrame:
    """Fetch current S&P 500 companies list"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url)[0]
        
        # Rename columns for consistency
        df = df.rename(columns={
            'Symbol': 'ticker',
            'Security': 'company_name',
            'GICS Sector': 'sector',
            'GICS Sub-Industry': 'industry',
        })
        
        # Clean ticker symbols
        df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)
        
        logger.info(f"Fetched {len(df)} S&P 500 companies")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list: {e}")
        raise

# =====================================================
# STEP 2: Comprehensive Info Data Extraction
# =====================================================

def safe_get_value(value):
    """Convert value to safe format, handling NaN and None"""
    if value is None:
        return None
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (list, dict)):
        return value
    return value

def fetch_comprehensive_info(ticker: str, company_data: dict) -> Dict[str, Any]:
    """
    Fetch and organize ALL info data from yfinance
    Returns a comprehensive dictionary with all available fields organized by category
    """
    ticker = ticker.upper()
    logger.info(f"Fetching comprehensive info for {ticker}")
    
    result = {
        'ticker': ticker,
        'fetch_timestamp': datetime.now().isoformat(),
        'fetch_status': 'pending',
        'sp500_data': {
            'company_name': company_data.get('company_name', ''),
            'sector': company_data.get('sector', ''),
            'industry': company_data.get('industry', ''),
        }
    }
    
    try:
        # Fetch info from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            result['fetch_status'] = 'no_data'
            logger.warning(f"No info data for {ticker}")
            return result
        
        # Store raw info
        result['raw_info'] = {k: safe_get_value(v) for k, v in info.items()}
        
        # Organize data into comprehensive categories
        result['organized_data'] = {
            # Company Information
            'company_details': {
                'longName': info.get('longName'),
                'shortName': info.get('shortName'),
                'symbol': info.get('symbol'),
                'website': info.get('website'),
                'industry': info.get('industry'),
                'industryKey': info.get('industryKey'),
                'industryDisp': info.get('industryDisp'),
                'sector': info.get('sector'),
                'sectorKey': info.get('sectorKey'),
                'sectorDisp': info.get('sectorDisp'),
                'longBusinessSummary': info.get('longBusinessSummary'),
                'fullTimeEmployees': info.get('fullTimeEmployees'),
                'companyOfficers': info.get('companyOfficers', []),
                'irWebsite': info.get('irWebsite'),
            },
            
            # Location Information
            'location': {
                'address1': info.get('address1'),
                'city': info.get('city'),
                'state': info.get('state'),
                'zip': info.get('zip'),
                'country': info.get('country'),
                'phone': info.get('phone'),
            },
            
            # Trading Information
            'trading': {
                'exchange': info.get('exchange'),
                'exchangeTimezoneName': info.get('exchangeTimezoneName'),
                'exchangeTimezoneShortName': info.get('exchangeTimezoneShortName'),
                'currency': info.get('currency'),
                'financialCurrency': info.get('financialCurrency'),
                'quoteType': info.get('quoteType'),
                'marketState': info.get('marketState'),
                'market': info.get('market'),
                'tradeable': info.get('tradeable'),
                'priceHint': info.get('priceHint'),
            },
            
            # Price Metrics
            'price_metrics': {
                'currentPrice': info.get('currentPrice'),
                'previousClose': info.get('previousClose'),
                'open': info.get('open'),
                'dayLow': info.get('dayLow'),
                'dayHigh': info.get('dayHigh'),
                'regularMarketPrice': info.get('regularMarketPrice'),
                'regularMarketPreviousClose': info.get('regularMarketPreviousClose'),
                'regularMarketOpen': info.get('regularMarketOpen'),
                'regularMarketDayLow': info.get('regularMarketDayLow'),
                'regularMarketDayHigh': info.get('regularMarketDayHigh'),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'bidSize': info.get('bidSize'),
                'askSize': info.get('askSize'),
            },
            
            # Volume Metrics
            'volume_metrics': {
                'volume': info.get('volume'),
                'regularMarketVolume': info.get('regularMarketVolume'),
                'averageVolume': info.get('averageVolume'),
                'averageVolume10days': info.get('averageVolume10days'),
                'averageDailyVolume10Day': info.get('averageDailyVolume10Day'),
                'averageDailyVolume3Month': info.get('averageDailyVolume3Month'),
            },
            
            # 52 Week Metrics
            'yearly_metrics': {
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekRange': info.get('fiftyTwoWeekRange'),
                '52WeekChange': info.get('52WeekChange'),
                'SandP52WeekChange': info.get('SandP52WeekChange'),
            },
            
            # Moving Averages
            'moving_averages': {
                'fiftyDayAverage': info.get('fiftyDayAverage'),
                'fiftyDayAverageChange': info.get('fiftyDayAverageChange'),
                'fiftyDayAverageChangePercent': info.get('fiftyDayAverageChangePercent'),
                'twoHundredDayAverage': info.get('twoHundredDayAverage'),
                'twoHundredDayAverageChange': info.get('twoHundredDayAverageChange'),
                'twoHundredDayAverageChangePercent': info.get('twoHundredDayAverageChangePercent'),
            },
            
            # Valuation Metrics
            'valuation': {
                'marketCap': info.get('marketCap'),
                'enterpriseValue': info.get('enterpriseValue'),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'trailingPegRatio': info.get('trailingPegRatio'),
                'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
                'priceToBook': info.get('priceToBook'),
                'enterpriseToRevenue': info.get('enterpriseToRevenue'),
                'enterpriseToEbitda': info.get('enterpriseToEbitda'),
                'bookValue': info.get('bookValue'),
            },
            
            # Financial Metrics
            'financials': {
                'totalRevenue': info.get('totalRevenue'),
                'revenuePerShare': info.get('revenuePerShare'),
                'revenueGrowth': info.get('revenueGrowth'),
                'grossProfits': info.get('grossProfits'),
                'grossMargins': info.get('grossMargins'),
                'operatingMargins': info.get('operatingMargins'),
                'profitMargins': info.get('profitMargins'),
                'ebitda': info.get('ebitda'),
                'ebitdaMargins': info.get('ebitdaMargins'),
                'netIncomeToCommon': info.get('netIncomeToCommon'),
                'earningsGrowth': info.get('earningsGrowth'),
                'earningsQuarterlyGrowth': info.get('earningsQuarterlyGrowth'),
            },
            
            # Earnings & EPS
            'earnings': {
                'trailingEps': info.get('trailingEps'),
                'forwardEps': info.get('forwardEps'),
                'epsTrailingTwelveMonths': info.get('epsTrailingTwelveMonths'),
                'epsForward': info.get('epsForward'),
            },
            
            # Returns & Profitability
            'returns': {
                'returnOnAssets': info.get('returnOnAssets'),
                'returnOnEquity': info.get('returnOnEquity'),
            },
            
            # Balance Sheet
            'balance_sheet': {
                'totalCash': info.get('totalCash'),
                'totalCashPerShare': info.get('totalCashPerShare'),
                'totalDebt': info.get('totalDebt'),
                'debtToEquity': info.get('debtToEquity'),
                'currentRatio': info.get('currentRatio'),
                'quickRatio': info.get('quickRatio'),
            },
            
            # Cash Flow
            'cash_flow': {
                'freeCashflow': info.get('freeCashflow'),
                'operatingCashflow': info.get('operatingCashflow'),
            },
            
            # Shares Information
            'shares': {
                'sharesOutstanding': info.get('sharesOutstanding'),
                'floatShares': info.get('floatShares'),
                'sharesShort': info.get('sharesShort'),
                'shortRatio': info.get('shortRatio'),
                'shortPercentOfFloat': info.get('shortPercentOfFloat'),
            },
            
            # Ownership
            'ownership': {
                'heldPercentInsiders': info.get('heldPercentInsiders'),
                'heldPercentInstitutions': info.get('heldPercentInstitutions'),
            },
            
            # Dividends
            'dividends': {
                'dividendRate': info.get('dividendRate'),
                'dividendYield': info.get('dividendYield'),
                'exDividendDate': info.get('exDividendDate'),
                'payoutRatio': info.get('payoutRatio'),
                'fiveYearAvgDividendYield': info.get('fiveYearAvgDividendYield'),
            },
            
            # Analyst Recommendations
            'analyst': {
                'targetHighPrice': info.get('targetHighPrice'),
                'targetLowPrice': info.get('targetLowPrice'),
                'targetMeanPrice': info.get('targetMeanPrice'),
                'targetMedianPrice': info.get('targetMedianPrice'),
                'recommendationMean': info.get('recommendationMean'),
                'recommendationKey': info.get('recommendationKey'),
                'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions'),
            },
            
            # Risk Metrics
            'risk': {
                'beta': info.get('beta'),
                'auditRisk': info.get('auditRisk'),
                'boardRisk': info.get('boardRisk'),
                'compensationRisk': info.get('compensationRisk'),
                'shareHolderRightsRisk': info.get('shareHolderRightsRisk'),
                'overallRisk': info.get('overallRisk'),
            },
        }
        
        result['fetch_status'] = 'success'
        result['info_field_count'] = len([v for v in info.values() if v is not None])
        
        logger.info(f"Successfully fetched {result['info_field_count']} non-null fields for {ticker}")
        
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        result['fetch_status'] = 'error'
        result['error'] = str(e)
    
    return result

# Copy the calculate_comprehensive_scores and build_comprehensive_graph functions
# from the original script (they remain the same)

# =====================================================
# STEP 3: Enhanced Scoring with All Metrics
# =====================================================

def calculate_comprehensive_scores(stocks: List[Dict]) -> List[Dict]:
    """
    Calculate comprehensive scores using all available metrics
    """
    records = []
    
    for stock in stocks:
        if stock['fetch_status'] != 'success':
            continue
        
        data = stock['organized_data']
        
        # Extract all relevant metrics for scoring
        record = {
            'ticker': stock['ticker'],
            'sector': stock['sp500_data']['sector'],
            'industry': stock['sp500_data']['industry'],
            
            # Valuation metrics
            'market_cap': data['valuation'].get('marketCap', 0) or 0,
            'pe_ratio': data['valuation'].get('trailingPE', 0) or 0,
            'forward_pe': data['valuation'].get('forwardPE', 0) or 0,
            'peg_ratio': data['valuation'].get('trailingPegRatio', 0) or 0,
            'price_to_book': data['valuation'].get('priceToBook', 0) or 0,
            'price_to_sales': data['valuation'].get('priceToSalesTrailing12Months', 0) or 0,
            'ev_to_ebitda': data['valuation'].get('enterpriseToEbitda', 0) or 0,
            
            # Profitability metrics
            'profit_margin': data['financials'].get('profitMargins', 0) or 0,
            'operating_margin': data['financials'].get('operatingMargins', 0) or 0,
            'gross_margin': data['financials'].get('grossMargins', 0) or 0,
            'ebitda_margin': data['financials'].get('ebitdaMargins', 0) or 0,
            'roe': data['returns'].get('returnOnEquity', 0) or 0,
            'roa': data['returns'].get('returnOnAssets', 0) or 0,
            
            # Growth metrics
            'revenue_growth': data['financials'].get('revenueGrowth', 0) or 0,
            'earnings_growth': data['financials'].get('earningsGrowth', 0) or 0,
            'earnings_quarterly_growth': data['financials'].get('earningsQuarterlyGrowth', 0) or 0,
            
            # Financial health
            'current_ratio': data['balance_sheet'].get('currentRatio', 0) or 0,
            'quick_ratio': data['balance_sheet'].get('quickRatio', 0) or 0,
            'debt_to_equity': data['balance_sheet'].get('debtToEquity', 0) or 0,
            'free_cashflow': data['cash_flow'].get('freeCashflow', 0) or 0,
            
            # Dividend metrics
            'dividend_yield': data['dividends'].get('dividendYield', 0) or 0,
            'payout_ratio': data['dividends'].get('payoutRatio', 0) or 0,
            
            # Risk metrics
            'beta': data['risk'].get('beta', 1) or 1,
            'overall_risk': data['risk'].get('overallRisk', 5) or 5,
            
            # Momentum metrics (price performance)
            '52_week_change': data['yearly_metrics'].get('52WeekChange', 0) or 0,
            '50_day_change': data['moving_averages'].get('fiftyDayAverageChangePercent', 0) or 0,
            '200_day_change': data['moving_averages'].get('twoHundredDayAverageChangePercent', 0) or 0,
            
            # Analyst sentiment
            'analyst_rating': data['analyst'].get('recommendationMean', 3) or 3,
            'analyst_count': data['analyst'].get('numberOfAnalystOpinions', 0) or 0,
            
            # Ownership
            'institutional_ownership': data['ownership'].get('heldPercentInstitutions', 0) or 0,
            'insider_ownership': data['ownership'].get('heldPercentInsiders', 0) or 0,
        }
        records.append(record)
    
    if not records:
        logger.warning("No valid records for scoring")
        return stocks
    
    df = pd.DataFrame(records)
    
    # Calculate comprehensive scores
    
    # 1. Value Score (30% weight)
    df['value_score'] = 0
    # PE ratio component (lower is better, but not negative)
    valid_pe = (df['pe_ratio'] > 0) & (df['pe_ratio'] < 100)
    df.loc[valid_pe, 'value_score'] += (1 - df.loc[valid_pe, 'pe_ratio'] / 50) * 30
    # Price to book component
    valid_pb = (df['price_to_book'] > 0) & (df['price_to_book'] < 10)
    df.loc[valid_pb, 'value_score'] += (1 - df.loc[valid_pb, 'price_to_book'] / 10) * 20
    # Price to sales component
    valid_ps = (df['price_to_sales'] > 0) & (df['price_to_sales'] < 10)
    df.loc[valid_ps, 'value_score'] += (1 - df.loc[valid_ps, 'price_to_sales'] / 10) * 20
    # EV/EBITDA component
    valid_ev = (df['ev_to_ebitda'] > 0) & (df['ev_to_ebitda'] < 30)
    df.loc[valid_ev, 'value_score'] += (1 - df.loc[valid_ev, 'ev_to_ebitda'] / 30) * 30
    
    # 2. Quality Score (25% weight)
    df['quality_score'] = 0
    df['quality_score'] += np.clip(df['roe'] * 50, 0, 25)
    df['quality_score'] += np.clip(df['roa'] * 100, 0, 25)
    df['quality_score'] += np.clip(df['profit_margin'] * 50, 0, 25)
    df['quality_score'] += np.clip(df['operating_margin'] * 50, 0, 25)
    
    # 3. Growth Score (20% weight)
    df['growth_score'] = 0
    df['growth_score'] += np.clip(df['revenue_growth'] * 50, -25, 50)
    df['growth_score'] += np.clip(df['earnings_growth'] * 50, -25, 50)
    
    # 4. Financial Health Score (15% weight)
    df['financial_score'] = 0
    # Current ratio (>1.5 is good)
    df['financial_score'] += np.where(df['current_ratio'] > 1.5, 25,
                                      np.where(df['current_ratio'] > 1, 15, 0))
    # Quick ratio (>1 is good)
    df['financial_score'] += np.where(df['quick_ratio'] > 1, 25,
                                      np.where(df['quick_ratio'] > 0.75, 15, 0))
    # Debt to equity (<1 is good)
    df['financial_score'] += np.where(df['debt_to_equity'] < 0.5, 25,
                                      np.where(df['debt_to_equity'] < 1, 15,
                                              np.where(df['debt_to_equity'] < 2, 5, 0)))
    # Free cash flow (positive is good)
    df['financial_score'] += np.where(df['free_cashflow'] > 0, 25, 0)
    
    # 5. Momentum Score (10% weight)
    df['momentum_score'] = 0
    df['momentum_score'] += np.clip(df['52_week_change'] * 25, -50, 50)
    df['momentum_score'] += np.clip(df['50_day_change'] * 25, -25, 25)
    df['momentum_score'] += np.clip(df['200_day_change'] * 25, -25, 25)
    
    # Analyst score bonus (adds up to 10 points)
    df['analyst_bonus'] = 0
    # Better analyst rating (1=strong buy, 5=strong sell)
    df['analyst_bonus'] += np.where(df['analyst_rating'] < 2, 5,
                                    np.where(df['analyst_rating'] < 2.5, 3, 0))
    # More analyst coverage
    df['analyst_bonus'] += np.where(df['analyst_count'] > 20, 5,
                                    np.where(df['analyst_count'] > 10, 3, 0))
    
    # Calculate composite score
    df['composite_score'] = (
        df['value_score'] * 0.30 +
        df['quality_score'] * 0.25 +
        df['growth_score'] * 0.20 +
        df['financial_score'] * 0.15 +
        df['momentum_score'] * 0.10 +
        df['analyst_bonus']
    )
    
    # Normalize to 0-100 scale
    if df['composite_score'].max() != df['composite_score'].min():
        df['composite_score'] = (df['composite_score'] - df['composite_score'].min()) / \
                                (df['composite_score'].max() - df['composite_score'].min()) * 100
    
    # Add percentile rankings
    df['percentile_overall'] = df['composite_score'].rank(pct=True) * 100
    df['percentile_sector'] = df.groupby('sector')['composite_score'].rank(pct=True) * 100
    df['percentile_industry'] = df.groupby('industry')['composite_score'].rank(pct=True) * 100
    
    # Merge scores back to original data
    score_columns = ['value_score', 'quality_score', 'growth_score', 'financial_score', 
                    'momentum_score', 'analyst_bonus', 'composite_score',
                    'percentile_overall', 'percentile_sector', 'percentile_industry']
    score_dict = df.set_index('ticker')[score_columns].to_dict('index')
    
    for stock in stocks:
        if stock['ticker'] in score_dict:
            stock['scores'] = score_dict[stock['ticker']]
    
    return stocks

# =====================================================
# STEP 4: Enhanced Knowledge Graph
# =====================================================

def build_comprehensive_graph(stocks: List[Dict]) -> nx.Graph:
    """
    Build a comprehensive knowledge graph using all available data
    """
    G = nx.Graph()
    
    # Add index node
    G.add_node('SP500', 
              node_type='index',
              name='S&P 500 Index',
              timestamp=datetime.now().isoformat())
    
    # Track sectors, industries, and other groupings
    sectors = {}
    industries = {}
    countries = {}
    exchanges = {}
    
    for stock in stocks:
        if stock['fetch_status'] != 'success':
            continue
        
        ticker = stock['ticker']
        data = stock['organized_data']
        sp500_data = stock['sp500_data']
        
        # Add sector node
        sector = sp500_data['sector']
        if sector and sector not in sectors:
            sector_id = f"SECTOR_{sector.replace(' ', '_').upper()}"
            sectors[sector] = sector_id
            G.add_node(sector_id,
                      node_type='sector',
                      name=sector)
            G.add_edge('SP500', sector_id, relationship='contains')
        
        # Add industry node
        industry = sp500_data['industry']
        if industry and industry not in industries:
            industry_id = f"INDUSTRY_{industry.replace(' ', '_').upper()}"
            industries[industry] = industry_id
            G.add_node(industry_id,
                      node_type='industry',
                      name=industry,
                      sector=sector)
            if sector in sectors:
                G.add_edge(sectors[sector], industry_id, relationship='contains')
        
        # Add country node
        country = data['location'].get('country')
        if country and country not in countries:
            country_id = f"COUNTRY_{country.replace(' ', '_').upper()}"
            countries[country] = country_id
            G.add_node(country_id,
                      node_type='country',
                      name=country)
            G.add_edge('SP500', country_id, relationship='includes_companies_from')
        
        # Add exchange node
        exchange = data['trading'].get('exchange')
        if exchange and exchange not in exchanges:
            exchange_id = f"EXCHANGE_{exchange}"
            exchanges[exchange] = exchange_id
            G.add_node(exchange_id,
                      node_type='exchange',
                      name=exchange)
            G.add_edge('SP500', exchange_id, relationship='trades_on')
        
        # Add company node with ALL attributes
        company_attrs = {
            'node_type': 'company',
            'name': data['company_details'].get('longName') or sp500_data['company_name'],
            'ticker': ticker,
            'website': data['company_details'].get('website'),
            'employees': data['company_details'].get('fullTimeEmployees'),
            'summary': data['company_details'].get('longBusinessSummary'),
            
            # Location
            'city': data['location'].get('city'),
            'state': data['location'].get('state'),
            'country': data['location'].get('country'),
            
            # Market data
            'market_cap': data['valuation'].get('marketCap'),
            'pe_ratio': data['valuation'].get('trailingPE'),
            'forward_pe': data['valuation'].get('forwardPE'),
            'price_to_book': data['valuation'].get('priceToBook'),
            'dividend_yield': data['dividends'].get('dividendYield'),
            'beta': data['risk'].get('beta'),
            
            # Financial metrics
            'revenue': data['financials'].get('totalRevenue'),
            'profit_margin': data['financials'].get('profitMargins'),
            'roe': data['returns'].get('returnOnEquity'),
            'roa': data['returns'].get('returnOnAssets'),
            'debt_to_equity': data['balance_sheet'].get('debtToEquity'),
            
            # Scores (if available)
            'composite_score': stock.get('scores', {}).get('composite_score'),
            'percentile_overall': stock.get('scores', {}).get('percentile_overall'),
            'percentile_sector': stock.get('scores', {}).get('percentile_sector'),
        }
        
        # Remove None values for cleaner graph
        company_attrs = {k: v for k, v in company_attrs.items() if v is not None}
        G.add_node(ticker, **company_attrs)
        
        # Add edges
        if sector in sectors:
            G.add_edge(ticker, sectors[sector], relationship='belongs_to')
        if industry in industries:
            G.add_edge(ticker, industries[industry], relationship='operates_in')
        if country in countries:
            G.add_edge(ticker, countries[country], relationship='headquartered_in')
        if exchange in exchanges:
            G.add_edge(ticker, exchanges[exchange], relationship='listed_on')
    
    # Add similarity edges based on metrics
    tickers = [s['ticker'] for s in stocks if s['fetch_status'] == 'success']
    
    # Find similar companies by market cap
    for i, ticker1 in enumerate(tickers):
        if ticker1 not in G:
            continue
        market_cap1 = G.nodes[ticker1].get('market_cap', 0)
        if not market_cap1:
            continue
            
        for ticker2 in tickers[i+1:]:
            if ticker2 not in G:
                continue
            market_cap2 = G.nodes[ticker2].get('market_cap', 0)
            if not market_cap2:
                continue
                
            # Similar market cap (within 20%)
            if 0.8 <= market_cap2/market_cap1 <= 1.25:
                G.add_edge(ticker1, ticker2, 
                          relationship='similar_market_cap',
                          weight=1 - abs(market_cap1 - market_cap2)/max(market_cap1, market_cap2))
    
    logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


# =====================================================
# MAIN PIPELINE WITH CHROMADB
# =====================================================

def ingest_sp500_to_chromadb(limit: int = None, workers: int = 8, reset: bool = False):
    """
    Main ingestion pipeline using ChromaDB for vector storage
    """
    logger.info("="*60)
    logger.info("Starting S&P 500 Ingestion with ChromaDB")
    logger.info("="*60)
    
    # Initialize ChromaDB
    chroma_manager = ChromaDBManager()
    
    if reset:
        logger.info("Resetting ChromaDB collections...")
        chroma_manager.reset_collections()
    
    # Step 1: Get S&P 500 list
    logger.info("Step 1: Fetching S&P 500 list")
    sp500_df = fetch_sp500_list()
    
    if limit:
        sp500_df = sp500_df.head(limit)
        logger.info(f"Limited to {limit} companies for testing")
    
    # Step 2: Fetch comprehensive info data
    logger.info(f"Step 2: Fetching comprehensive info for {len(sp500_df)} stocks")
    stocks = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        
        for _, row in sp500_df.iterrows():
            future = executor.submit(fetch_comprehensive_info, row['ticker'], row.to_dict())
            futures[future] = row['ticker']
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching data"):
            try:
                result = future.result(timeout=30)
                stocks.append(result)
            except Exception as e:
                ticker = futures[future]
                logger.error(f"Failed to fetch {ticker}: {e}")
                stocks.append({
                    'ticker': ticker,
                    'fetch_status': 'error',
                    'error': str(e)
                })
    
    # Step 3: Calculate comprehensive scores
    logger.info("Step 3: Calculating comprehensive scores")
    stocks = calculate_comprehensive_scores(stocks)
    
    # Step 4: Build knowledge graph
    logger.info("Step 4: Building knowledge graph")
    graph = build_comprehensive_graph(stocks)
    
    # Step 5: Add data to ChromaDB
    logger.info("Step 5: Adding data to ChromaDB")
    chroma_manager.add_company_data(stocks)
    chroma_manager.add_sector_summaries(stocks)
    
    # Step 6: Save supplementary data
    logger.info("Step 6: Saving supplementary data")
    
    # Save raw data
    with open(STORE_DIR / "comprehensive_stocks_data.json", "w") as f:
        json.dump(stocks, f, indent=2, default=str)
    
    # Save graph
    with open(STORE_DIR / "comprehensive_knowledge_graph.pkl", "wb") as f:
        pickle.dump(graph, f)
    
    # Generate summary
    stats = chroma_manager.get_statistics()
    success_count = sum(1 for s in stocks if s['fetch_status'] == 'success')
    error_count = sum(1 for s in stocks if s['fetch_status'] == 'error')
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_companies': len(stocks),
        'successful_fetches': success_count,
        'failed_fetches': error_count,
        'graph_nodes': graph.number_of_nodes(),
        'graph_edges': graph.number_of_edges(),
        'chromadb_collections': stats,
        'sectors': len(set(s['sp500_data']['sector'] for s in stocks if s['fetch_status'] == 'success')),
        'industries': len(set(s['sp500_data']['industry'] for s in stocks if s['fetch_status'] == 'success')),
    }
    
    with open(STORE_DIR / "ingestion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info("="*60)
    logger.info("INGESTION COMPLETE!")
    logger.info(f"Successfully processed {success_count}/{len(stocks)} companies")
    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    logger.info("ChromaDB Collections:")
    for name, stat in stats.items():
        logger.info(f"  {name}: {stat['count']} documents")
    logger.info(f"Data saved to: {STORE_DIR}")
    logger.info("="*60)
    
    return {
        'stocks': stocks,
        'graph': graph,
        'chroma_manager': chroma_manager,
        'summary': summary
    }


# =====================================================
# QUERY INTERFACE
# =====================================================

def query_chromadb(query: str, collection: str = 'companies', n_results: int = 5,
                  filters: Dict = None):
    """Query ChromaDB collections"""
    chroma_manager = ChromaDBManager()
    
    results = chroma_manager.query(
        query_text=query,
        collection_name=collection,
        n_results=n_results,
        where=filters
    )
    
    if results:
        print(f"\nQuery: {query}")
        print(f"Collection: {collection}")
        if filters:
            print(f"Filters: {filters}")
        print("="*60)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n{i}. {metadata.get('ticker', 'N/A')} - {metadata.get('company_name', 'N/A')}")
            print(f"   Sector: {metadata.get('sector', 'N/A')}")
            print(f"   Distance: {distance:.4f}")
            print(f"   Document Type: {metadata.get('document_type', 'N/A')}")
            print(f"   Preview: {doc[:200]}...")
    
    return results


def advanced_search(sector: str = None, min_score: float = None, 
                   max_pe: float = None, min_market_cap: float = None):
    """Advanced search with filters"""
    chroma_manager = ChromaDBManager()
    
    # Build where clause
    where_conditions = {}
    if sector:
        where_conditions['sector'] = sector
    if min_score:
        where_conditions['composite_score'] = {'$gte': min_score}
    if max_pe:
        where_conditions['pe_ratio'] = {'$lte': max_pe}
    if min_market_cap:
        where_conditions['market_cap'] = {'$gte': min_market_cap}
    
    # Query with filters
    query_text = "high performing companies with strong fundamentals"
    
    results = chroma_manager.query(
        query_text=query_text,
        collection_name='companies',
        n_results=10,
        where=where_conditions if where_conditions else None
    )
    
    return results


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='S&P 500 ChromaDB Ingestion')
    parser.add_argument('--ingest', action='store_true', help='Run ingestion pipeline')
    parser.add_argument('--limit', type=int, help='Limit number of companies')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--reset', action='store_true', help='Reset ChromaDB collections')
    parser.add_argument('--query', type=str, help='Query ChromaDB')
    parser.add_argument('--collection', type=str, default='companies', 
                       choices=['companies', 'financials', 'trading', 'sectors'],
                       help='Collection to query')
    parser.add_argument('--search', action='store_true', help='Advanced search')
    parser.add_argument('--sector', type=str, help='Filter by sector')
    parser.add_argument('--min-score', type=float, help='Minimum composite score')
    parser.add_argument('--max-pe', type=float, help='Maximum P/E ratio')
    parser.add_argument('--min-cap', type=float, help='Minimum market cap')
    
    args = parser.parse_args()
    
    if args.ingest:
        # Run ingestion
        result = ingest_sp500_to_chromadb(
            limit=args.limit,
            workers=args.workers,
            reset=args.reset
        )
        
    elif args.query:
        # Query ChromaDB
        filters = {}
        if args.sector:
            filters['sector'] = args.sector
        
        query_chromadb(
            query=args.query,
            collection=args.collection,
            filters=filters if filters else None
        )
        
    elif args.search:
        # Advanced search
        results = advanced_search(
            sector=args.sector,
            min_score=args.min_score,
            max_pe=args.max_pe,
            min_market_cap=args.min_cap
        )
        
        if results and results['documents'][0]:
            print("\nAdvanced Search Results:")
            print("="*60)
            for i, metadata in enumerate(results['metadatas'][0], 1):
                print(f"{i}. {metadata.get('ticker')} - {metadata.get('company_name')}")
                print(f"   Sector: {metadata.get('sector')}")
                print(f"   Score: {metadata.get('composite_score', 0):.2f}")
                print(f"   P/E: {metadata.get('pe_ratio', 'N/A')}")
                print()
    
    else:
        print("Usage examples:")
        print("  # Ingest data")
        print("  python chromadb_sp500_ingest.py --ingest --limit 10")
        print("  python chromadb_sp500_ingest.py --ingest --reset  # Full ingestion with reset")
        print()
        print("  # Query data")
        print("  python chromadb_sp500_ingest.py --query 'technology companies with high growth'")
        print("  python chromadb_sp500_ingest.py --query 'dividend stocks' --collection financials")
        print()
        print("  # Advanced search")
        print("  python chromadb_sp500_ingest.py --search --sector Technology --min-score 70")
        print("  python chromadb_sp500_ingest.py --search --max-pe 20 --min-cap 100000000000")