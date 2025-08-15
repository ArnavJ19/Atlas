# 📈 ATLAS

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/React-18.0+-61dafb.svg" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-5.0+-blue.svg" alt="TypeScript">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

<div align="center">
  <h3>🚀 Advanced Stock Market Analytics with RAG-Powered AI Assistant</h3>
  <p>A comprehensive S&P 500 dashboard combining real-time market data, AI-driven insights, and knowledge graph visualization for intelligent investment analysis.</p>
</div>

---

## ✨ Features

### 🤖 AI-Powered Intelligence
- **RAG System**: Natural language queries about stocks using ChromaDB vector database
- **Smart Recommendations**: AI-generated insights based on comprehensive market data
- **Semantic Search**: Find stocks using conversational queries
- **Knowledge Graph**: NetworkX-powered relationship mapping between companies

### 📊 Real-Time Market Data
- **Live Stock Prices**: Real-time updates via WebSocket connections
- **Market Overview**: S&P 500 index and sector performance tracking
- **Historical Analysis**: Interactive charts with multiple timeframes
- **Volume Analysis**: Trading volume patterns and trends

### 🔍 Advanced Analytics
- **Stock Screener**: Filter by market cap, P/E ratio, dividends, sectors, and AI scores
- **Portfolio Analyzer**: Build and analyze custom portfolios with risk metrics
- **Company Deep Dive**: Comprehensive company analysis with financial metrics
- **Network Visualization**: Interactive D3.js knowledge graph

### 💼 Investment Tools
- **Composite Scoring**: Multi-factor scoring system (value, quality, growth, momentum)
- **Sector Analysis**: Compare performance across different sectors
- **Risk Assessment**: Beta, volatility, and Sharpe ratio calculations
- **Peer Comparison**: Find similar companies using graph relationships

## 🛠️ Tech Stack

### Backend
- **FastAPI** - High-performance API framework
- **ChromaDB** - Vector database for semantic search
- **NetworkX** - Graph analysis and visualization
- **yfinance** - Real-time market data
- **LangChain** - LLM orchestration (optional)
- **WebSockets** - Real-time communication

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Chart.js** - Interactive charts
- **D3.js** - Network visualization
- **React Query** - Data fetching and caching
- **Framer Motion** - Smooth animations

## 📋 Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- Git

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sp500-ai-dashboard.git
cd sp500-ai-dashboard
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data ingestion (first time setup)
python chromadb_sp500_ingest.py --ingest --reset --limit 50  # Test with 50 companies
# OR for full S&P 500
python chromadb_sp500_ingest.py --ingest --reset

# Start the backend server
python rag_backend.py
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
# Open new terminal and navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📁 Project Structure

```
sp500-ai-dashboard/
├── backend/
│   ├── rag_backend.py              # FastAPI server with RAG
│   ├── chromadb_sp500_ingest.py    # Data ingestion pipeline
│   ├── chromadb_kg_explorer.py     # Knowledge graph exploration
│   ├── requirements.txt            # Python dependencies
│   └── store/                      # Data storage
│       ├── chromadb/               # Vector database
│       └── *.pkl, *.json          # Graph and metadata
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main application
│   │   ├── components/            # React components
│   │   ├── services/              # API services
│   │   ├── hooks/                 # Custom React hooks
│   │   └── types/                 # TypeScript definitions
│   ├── package.json               # Node dependencies
│   ├── vite.config.ts            # Vite configuration
│   └── tailwind.config.js        # Tailwind CSS config
│
└── README.md
```

## 🔌 API Endpoints

### RAG & AI
- `POST /api/rag/query` - Semantic search with RAG
- `POST /api/rag/chat` - AI chat with conversation memory

### Market Data
- `GET /api/stock/{ticker}` - Individual stock data
- `GET /api/market/overview` - Market overview
- `GET /api/market/sectors` - Sector performance

### Analytics
- `POST /api/screener` - Advanced stock screening
- `POST /api/portfolio/analyze` - Portfolio analysis
- `GET /api/graph/{ticker}` - Graph insights
- `GET /api/graph/network` - Network visualization data

### Real-Time
- `WS /ws/{client_id}` - WebSocket for live updates

## 💡 Usage Examples

### AI Assistant Queries
- "Show me technology stocks with P/E ratio under 20"
- "Find high dividend yield companies in the healthcare sector"
- "What are the top performing stocks this month?"
- "Compare AAPL and MSFT financial metrics"

### Stock Screener
Filter stocks by:
- Market capitalization
- P/E ratio
- Dividend yield
- AI composite score
- Sector/Industry

### Portfolio Analysis
1. Add stocks to your portfolio
2. Set allocation weights
3. Analyze risk/return metrics
4. View performance visualization

## 🔧 Configuration

### Environment Variables

Create `.env` file in backend directory:
```env
# Optional - for LLM features
OPENAI_API_KEY=your_api_key_here

# Optional - for caching
REDIS_URL=redis://localhost:6379
```

Create `.env` file in frontend directory:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## 📊 Data Sources

- **Market Data**: Yahoo Finance (yfinance)
- **Company List**: Wikipedia S&P 500 list
- **Vector Storage**: ChromaDB
- **Graph Data**: NetworkX

## 🚢 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Manual Deployment

**Backend (using Gunicorn):**
```bash
gunicorn rag_backend:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend (production build):**
```bash
npm run build
# Serve with any static file server
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm run test
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://react.dev/) for the frontend framework


