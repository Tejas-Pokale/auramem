"""Generate distractor sessions once, then freeze them (PLAN.md section 10, corpus v2 step 3).

    uv run python -m evals.custom.generate_distractors --count 180

Writes data/distractors/distractors_v1.jsonl. The file is reviewed and committed; it is
never regenerated during evaluation. Sessions contain general tech Q&A and small talk with
no personal facts about the user, so they add realistic retrieval noise without
contradicting the fact corpus. Sessions that leak personal details are filtered out.
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from memory.config import get_settings

PROMPT_VERSION = "distractor_v1"
MODEL = "claude-haiku-5-5"
EFFORT = "low"
CONCURRENCY = 8
OUT = Path(__file__).parent / "data" / "distractors" / "distractors_v1.jsonl"

PROMPT = """Write a realistic chat between a user and an AI assistant about: {topic}

Rules:
- {exchanges} exchanges: the user asks, the assistant answers; follow-ups build on earlier answers.
- The user asks general questions only. The user must NOT reveal anything about themselves:
  no name, job, employer, school, location, family, health, finances, possessions, plans,
  dates of personal events, or personal preferences. Phrase questions impersonally
  (e.g. "How does X work?", "What's the difference between X and Y?").
- Assistant answers are accurate and specific, 60-150 words each, plain prose or short lists.
- No greetings or sign-offs."""

TECH_TOPICS = [
    "Python list comprehensions vs generator expressions",
    "Python asyncio event loop basics",
    "Python type hints and mypy strict mode",
    "Python virtual environments and uv",
    "Python decorators",
    "Python dataclasses vs Pydantic models",
    "Python GIL and threading",
    "Python packaging with pyproject.toml",
    "pytest fixtures and parametrization",
    "JavaScript closures",
    "JavaScript promises and async/await",
    "TypeScript generics",
    "TypeScript discriminated unions",
    "React useEffect pitfalls",
    "React server components",
    "Next.js app router data fetching",
    "CSS grid vs flexbox",
    "Tailwind CSS utility classes",
    "Node.js event loop",
    "Express middleware ordering",
    "REST vs GraphQL tradeoffs",
    "HTTP caching headers",
    "WebSockets vs Server-Sent Events",
    "OAuth 2.0 authorization code flow",
    "JWT pitfalls",
    "SQL joins explained",
    "SQL window functions",
    "database indexing strategies",
    "PostgreSQL VACUUM and bloat",
    "PostgreSQL JSONB queries",
    "MySQL vs PostgreSQL differences",
    "MongoDB schema design",
    "Redis data structures",
    "Redis as a queue",
    "database transactions and isolation levels",
    "CAP theorem",
    "consistent hashing",
    "message queues: Kafka vs RabbitMQ",
    "event sourcing basics",
    "microservices vs monoliths",
    "API rate limiting algorithms",
    "load balancing strategies",
    "Docker multi-stage builds",
    "Docker networking",
    "Docker volumes vs bind mounts",
    "Kubernetes pods vs deployments",
    "Kubernetes services and ingress",
    "Kubernetes ConfigMaps and Secrets",
    "Kubernetes resource requests and limits",
    "Kubernetes horizontal pod autoscaling",
    "Kubernetes GPU scheduling in general",
    "Helm chart structure",
    "Terraform state management",
    "GitHub Actions caching",
    "CI/CD deployment strategies: blue-green vs canary",
    "Prometheus and Grafana monitoring",
    "OpenTelemetry tracing",
    "structured logging",
    "Linux file permissions",
    "Linux process signals",
    "bash scripting pitfalls",
    "SSH key management",
    "Git rebase vs merge",
    "Git bisect",
    "semantic versioning",
    "AWS S3 storage classes",
    "AWS Lambda cold starts",
    "Google Cloud Run vs GKE",
    "Azure Functions triggers",
    "serverless pricing models",
    "CDN basics",
    "DNS resolution",
    "TLS handshake",
    "linear regression assumptions",
    "logistic regression vs decision trees",
    "overfitting and regularization",
    "cross-validation",
    "precision, recall and F1",
    "ROC curves and AUC",
    "gradient descent variants",
    "batch normalization",
    "dropout",
    "convolutional neural networks",
    "recurrent neural networks and LSTMs",
    "the transformer architecture at a high level",
    "self-attention complexity",
    "positional encodings compared",
    "byte-level BPE tokenization",
    "word embeddings vs contextual embeddings",
    "cosine similarity vs dot product",
    "approximate nearest neighbor search: HNSW",
    "vector database comparison",
    "BM25 ranking",
    "hybrid search and reciprocal rank fusion",
    "cross-encoder rerankers",
    "chunking strategies for RAG",
    "evaluating RAG systems",
    "prompt engineering techniques",
    "few-shot vs zero-shot prompting",
    "LLM temperature and sampling",
    "LLM context windows and long context",
    "LLM hallucinations",
    "function calling in LLM APIs",
    "structured outputs with JSON schema",
    "AI agent design patterns",
    "LangGraph state machines",
    "model quantization: GPTQ vs AWQ vs GGUF",
    "LoRA and QLoRA explained",
    "RLHF overview",
    "DPO vs PPO",
    "knowledge distillation",
    "mixture of experts models",
    "vLLM paged attention",
    "KV cache",
    "speculative decoding",
    "batching in LLM inference",
    "model serving latency metrics",
    "MLflow experiment tracking",
    "data drift detection",
    "feature stores",
    "Apache Spark partitions",
    "Delta Lake time travel",
    "Airflow DAG design",
    "data warehouse vs data lake",
    "star schema modeling",
    "pandas groupby performance",
    "NumPy broadcasting",
    "scikit-learn pipelines",
    "PyTorch autograd",
    "PyTorch DataLoader workers",
    "mixed precision training",
    "CUDA out-of-memory errors in general",
    "GPU memory for training vs inference",
    "Flutter state management options",
    "Dart null safety",
    "React Native vs Flutter",
    "Android activity lifecycle",
    "Kotlin coroutines",
    "Java streams API",
    "Java garbage collection",
    "Spring Boot basics",
    "Go goroutines and channels",
    "Rust ownership",
    "C++ smart pointers",
    "big-O notation",
    "binary search edge cases",
    "dynamic programming approach",
    "graph algorithms: BFS vs DFS",
    "hash tables collision handling",
    "regex lookaheads",
    "Unicode and UTF-8",
    "floating point precision",
    "OCR accuracy metrics in general",
    "speech-to-text evaluation metrics",
    "IoT protocols: MQTT",
    "ESP32 vs Raspberry Pi",
    "edge AI deployment",
]

GENERAL_TOPICS = [
    "how to brew good coffee",
    "basics of houseplant care",
    "how rainbows form",
    "why the sky is blue",
    "how vaccines train the immune system",
    "how compound interest works",
    "how to read a nutrition label",
    "the history of the printing press",
    "how airplanes stay in the air",
    "how tides work",
    "the rules of cricket's DRS system",
    "how chess openings are named",
    "tips for public speaking",
    "how to write a clear email",
    "the Pomodoro technique",
    "how noise-cancelling headphones work",
    "how electric cars charge",
    "how solar panels work",
    "how monsoons form",
    "the water cycle",
    "basics of photography exposure",
    "how to learn a new language efficiently",
    "how memory works in the brain",
    "how sleep cycles work",
    "stretching for desk workers",
    "how to make a simple dal",
    "the history of chai",
    "how bread rises",
    "differences between tea varieties",
    "how to plan a budget-friendly trip in general",
    "how trains are scheduled",
    "how elections are counted",
    "how the stock market works at a high level",
    "how credit scores work in general",
    "the plot structure of classic novels",
    "how movies are color graded",
    "music theory: major vs minor keys",
    "how 3D printers work",
    "how black holes form",
    "how GPS works",
]


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str


class DistractorSession(BaseModel):
    turns: list[Turn]


BLOCKED_TERMS = re.compile(
    r"\b(teju|tejas|pokale|wardha|yavatmal|arvi|dbatu|meditab|jhex|optum|crave infotech|zycus|"
    r"comcare|redconnect|recyclens|nagpur|tcs prime)\b",
    re.IGNORECASE,
)
PERSONAL = re.compile(
    r"\bmy (mother|mom|father|dad|sister|brother|wife|husband|partner|family|salary|job|company|"
    r"boss|manager|college|university|laptop|cgpa|gpa|degree|internship|health|weight|city|team)\b"
    r"|\bI (work|worked|live|lived|study|studied|joined|graduated|am from)\b"
    r"|\bI'm (from|working|studying|a |an )",
    re.IGNORECASE,
)


def leaks_personal(session: DistractorSession) -> str | None:
    for turn in session.turns:
        if m := BLOCKED_TERMS.search(turn.text):
            return f"blocked term {m.group(0)!r}"
        if turn.role == "user" and (m := PERSONAL.search(turn.text)):
            return f"personal phrase {m.group(0)!r}"
    if not session.turns or session.turns[0].role != "user":
        return "does not start with a user turn"
    if any(a.role == b.role for a, b in zip(session.turns, session.turns[1:], strict=False)):
        return "roles do not alternate"
    return None


async def generate(count: int) -> None:
    topics = (TECH_TOPICS + GENERAL_TOPICS)[:count]
    if len(topics) < count:
        raise SystemExit(f"only {len(topics)} topics available")
    settings = get_settings()
    llm = ChatAnthropic(
        model=MODEL, api_key=settings.anthropic_api_key, max_tokens=4096, effort=EFFORT
    ).with_structured_output(DistractorSession, method="json_schema")
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(i: int, topic: str) -> dict[str, object] | None:
        exchanges = 3 + i % 2
        async with sem:
            result = await llm.ainvoke(PROMPT.format(topic=topic, exchanges=exchanges))
        session = DistractorSession.model_validate(result)
        if reason := leaks_personal(session):
            print(f"  rejected {topic!r}: {reason}")
            return None
        return {
            "topic": topic,
            "turns": [t.model_dump() for t in session.turns],
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
        }

    results = await asyncio.gather(*(one(i, t) for i, t in enumerate(topics)))
    kept = [r for r in results if r is not None]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for n, record in enumerate(kept, start=1):
            f.write(json.dumps({"distractor_id": f"D{n:03d}", **record}, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}/{len(topics)} sessions -> {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=len(TECH_TOPICS) + len(GENERAL_TOPICS))
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(
            f"{OUT} exists; distractors are frozen. Delete it deliberately to regenerate."
        )
    asyncio.run(generate(args.count))


if __name__ == "__main__":
    main()
