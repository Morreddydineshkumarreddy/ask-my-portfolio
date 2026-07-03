# 💬 Ask My Portfolio — RAG Stock Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you ask natural-language questions about your stock portfolio. It fetches real-time market data, builds a vector knowledge base, and uses an LLM to give accurate, grounded answers.

> *"Is AAPL overbought right now?"*  
> *"Which of my stocks had the best month?"*  
> *"What's the volatility on TSLA compared to MSFT?"*

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│   yfinance API      │  ← Fetches real-time OHLCV + fundamentals
└────────┬────────────┘
         │  Stock data chunks (natural language)
         ▼
┌─────────────────────┐
│  Sentence-BERT      │  ← Embeds chunks into vector space
│  (all-MiniLM-L6-v2) │
└────────┬────────────┘
         │  In-memory vector store
         ▼
┌─────────────────────┐
│  Cosine Similarity  │  ← Retrieves top-k relevant chunks
│  Retriever          │
└────────┬────────────┘
         │  Context + Question
         ▼
┌─────────────────────┐
│  OpenAI GPT         │  ← Generates grounded answer
│  (gpt-4o-mini)      │
└─────────────────────┘
         │
         ▼
    Answer to User
```

---

## ✨ Features

- **Real-time data** — pulls fresh prices, returns, volume, RSI, moving averages on every run
- **Natural language interface** — no need to know tickers or formulas; just ask
- **RAG architecture** — answers are grounded in real data, not hallucinated
- **Lightweight** — no external vector DB; runs entirely in memory
- **Configurable** — swap in any set of tickers or time window via CLI flags

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Data fetching | `yfinance` |
| Embeddings | `sentence-transformers` (MiniLM) |
| Vector similarity | `numpy` cosine similarity |
| LLM | `openai` (GPT-4o-mini) |
| CLI | Python `argparse` |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Morreddydineshkumarreddy/ask-my-portfolio.git
cd ask-my-portfolio
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 3. Run

```bash
python main.py
```

---

## ⚙️ Options

```bash
python main.py --tickers AAPL MSFT TSLA --days 60 --model gpt-4o-mini
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers` | AAPL MSFT GOOGL TSLA AMZN | Stocks to load |
| `--days` | 90 | Historical window |
| `--model` | gpt-4o-mini | OpenAI model |

---

## 💡 Example Session

```
You: Is AAPL trending up or down?
🤖 Assistant: Based on the latest data, AAPL is currently in an uptrend — 
the price of $189.30 is above its 20-day moving average of $184.12. 
The 1-month return is +6.2%.

You: Which stock is the most volatile?
🤖 Assistant: TSLA has the highest annualised volatility at 62.4%, 
compared to AAPL (28.1%), MSFT (24.3%), GOOGL (29.7%), and AMZN (35.2%).

You: Is any stock overbought?
🤖 Assistant: NVDA's RSI is currently 74.1 — this is in overbought territory 
(above 70), which can sometimes signal a short-term pullback. All other 
stocks in your portfolio have neutral RSI readings.
```

---

## 📁 Project Structure

```
ask-my-portfolio/
├── main.py              # CLI entry point & chat loop
├── requirements.txt
├── .env.example
└── src/
    ├── data_fetcher.py  # yfinance data → natural language chunks
    ├── retriever.py     # Sentence-BERT embeddings + cosine retrieval
    └── llm.py           # OpenAI RAG prompt + response
```

---

## 🔮 Possible Extensions

- [ ] Add a Streamlit web UI
- [ ] Support news sentiment as an additional knowledge source
- [ ] Persist vector store to disk (FAISS / ChromaDB)
- [ ] Add portfolio P&L tracking
- [ ] Integrate with a free LLM (Llama via Ollama) to avoid API costs

---

## 📚 What I Learned

- How RAG (Retrieval-Augmented Generation) pipelines work end-to-end
- Generating and comparing sentence embeddings with `sentence-transformers`
- Grounding LLM responses in real data to avoid hallucinations
- Building clean CLI tools with argparse and ANSI formatting

---

*Built by [Dinesh Kumar Reddy M](https://github.com/Morreddydineshkumarreddy)*
