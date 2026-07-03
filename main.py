"""
Ask My Portfolio — RAG Chatbot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author : Dinesh Kumar Reddy M
GitHub : https://github.com/Morreddydineshkumarreddy

Ask natural-language questions about your stock portfolio.
The bot fetches real-time data, embeds it, and uses an LLM to answer.

Usage:
    python main.py
    python main.py --tickers AAPL MSFT TSLA --days 60
"""

import argparse
import sys
import os
from src.data_fetcher import fetch_portfolio_data
from src.retriever    import VectorStore
from src.llm          import answer_question

# ── ANSI colours for a clean CLI experience ────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

BANNER = f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════╗
║           💼  Ask My Portfolio  📈                  ║
║      RAG-Powered Stock Portfolio Chatbot             ║
╚══════════════════════════════════════════════════════╝
{RESET}"""

EXAMPLE_QUESTIONS = [
    "Is AAPL trending up or down?",
    "Which stock has the highest volatility?",
    "What is the RSI for TSLA?",
    "How did MSFT perform over the last month?",
    "Which stock had the best 3-month return?",
    "Is any stock overbought or oversold?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Ask My Portfolio — RAG Stock Chatbot")
    parser.add_argument(
        "--tickers", nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
        help="Stock tickers to load (default: AAPL MSFT GOOGL TSLA AMZN)"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Historical window in days (default: 90)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    return parser.parse_args()


def check_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"{YELLOW}⚠️  OPENAI_API_KEY not set.")
        print("   Run:  export OPENAI_API_KEY='sk-...'")
        print(f"   Then re-run the script.{RESET}")
        sys.exit(1)


def print_examples():
    print(f"\n{CYAN}💡 Example questions you can ask:{RESET}")
    for q in EXAMPLE_QUESTIONS:
        print(f"   • {q}")
    print()


def main():
    args = parse_args()
    check_api_key()

    print(BANNER)
    print(f"{BOLD}Portfolio:{RESET} {', '.join(args.tickers)}")
    print(f"{BOLD}Window   :{RESET} Last {args.days} days\n")

    # ── Step 1: Fetch real-time stock data ─────────────────────────────────────
    print(f"{CYAN}{'─'*54}")
    print("  Step 1 — Fetching portfolio data...")
    print(f"{'─'*54}{RESET}")
    portfolio_data = fetch_portfolio_data(args.tickers)

    if not portfolio_data:
        print("❌ No data fetched. Check your ticker symbols and internet connection.")
        sys.exit(1)

    loaded = [d["ticker"] for d in portfolio_data]
    print(f"  ✅ Loaded: {', '.join(loaded)}\n")

    # ── Step 2: Build vector store ─────────────────────────────────────────────
    print(f"{CYAN}{'─'*54}")
    print("  Step 2 — Building knowledge base...")
    print(f"{'─'*54}{RESET}")
    store = VectorStore()
    store.add_documents(portfolio_data)

    # ── Step 3: Chat loop ──────────────────────────────────────────────────────
    print(f"{GREEN}{BOLD}✅ Ready! Ask me anything about your portfolio.{RESET}")
    print_examples()
    print(f"   Type {YELLOW}'quit'{RESET} or {YELLOW}'exit'{RESET} to stop.\n")

    while True:
        try:
            question = input(f"{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{CYAN}Goodbye! 👋{RESET}")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print(f"{CYAN}Goodbye! 👋{RESET}")
            break

        if question.lower() in ("help", "examples", "?"):
            print_examples()
            continue

        # Retrieve relevant chunks
        context = store.retrieve(question, top_k=6)

        if not context:
            print(f"{YELLOW}⚠️  No relevant data found for that question.{RESET}\n")
            continue

        # Generate answer
        print(f"\n{CYAN}🤖 Assistant:{RESET} ", end="", flush=True)
        try:
            answer = answer_question(question, context, model=args.model)
            print(answer)
        except Exception as e:
            print(f"{YELLOW}Error calling LLM: {e}{RESET}")
        print()


if __name__ == "__main__":
    main()
