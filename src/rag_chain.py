# src/rag_chain.py

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from src.prompts import rag_prompt
from src.groq_client import get_llm


def format_context(chunks):
    blocks = []

    for i, c in enumerate(chunks, start=1):
        blocks.append(f"""
[SOURCE {i}]
Document: {c.get("doc_name", "unknown")}
Page: {c.get("page", "N/A")}
Source type: {c.get("source_type", "case_docs")}
Document type: {c.get("doc_type", "unknown")}
Chunk type: {c.get("chunk_type", "unknown")}

Text:
{c.get("text", "")}
""")

    return "\n\n".join(blocks)


def build_messages_from_history(history):
    messages = []

    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))

    return messages


def answer_with_langchain(question, chunks, route, chat_history=None):
    if chat_history is None:
        chat_history = []

    llm = get_llm()

    context = format_context(chunks)
    lc_history = build_messages_from_history(chat_history)

    chain = rag_prompt | llm

    response = chain.invoke({
        "question": question,
        "route": route,
        "context": context,
        "chat_history": lc_history,
    })

    return response.content