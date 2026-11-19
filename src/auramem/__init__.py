"""auramem: long-term memory for LLM apps.

    from auramem import AuraMem

    with AuraMem() as mem:                       # settings from .env / environment
        reply = mem.chat("user-42", "I just moved to Leeds.")
        reply = mem.chat("user-42", "Where do I live now?")
        print(reply.text)

Bring your own LLM instead of the built-in answerer:

    recall = mem.recall("user-42", question)     # memory for this message, nothing answered
    answer = my_llm(recall.as_prompt() + "\\n\\n" + question)
    mem.remember("user-42", question, answer)    # store the exchange

Async code uses ``AsyncAuraMem`` with the same methods (``async with AsyncAuraMem() as mem``).
Needs Neo4j, Ollama (EmbeddingGemma), and an Anthropic API key; see ``docker-compose.yml``.
"""

from auramem._async import AsyncAuraMem, Recall
from auramem._sync import AuraMem
from memory.models import DeepTrace, Fact, ForgetItem, ForgetPreview, ForgetReport, Reply

__all__ = [
    "AsyncAuraMem",
    "AuraMem",
    "DeepTrace",
    "Fact",
    "ForgetItem",
    "ForgetPreview",
    "ForgetReport",
    "Recall",
    "Reply",
]
__version__ = "0.1.0"
