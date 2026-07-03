"""
llm.py
Sends retrieved context + user question to an LLM and returns the answer.
Supports OpenAI GPT models (gpt-3.5-turbo / gpt-4o-mini).
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a knowledgeable financial assistant for a personal stock portfolio.
You answer questions based ONLY on the provided context about the user's stocks.
Be concise, factual, and clear. Use bullet points when listing multiple data points.
If the context doesn't contain enough information to answer, say so honestly.
Never fabricate financial data or make specific investment recommendations."""


def answer_question(question: str, context_chunks: list[str],
                    model: str = "gpt-4o-mini") -> str:
    """
    Build a RAG prompt and call the LLM.

    Args:
        question:      User's natural language question.
        context_chunks: Relevant chunks retrieved from the vector store.
        model:         OpenAI model name to use.

    Returns:
        The LLM's answer as a string.
    """
    context = "\n".join(f"- {chunk}" for chunk in context_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context (real-time portfolio data):\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()
