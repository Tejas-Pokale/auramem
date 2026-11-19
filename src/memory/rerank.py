"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3 (PLAN.md section 6, step 5).

Always rerank against the ORIGINAL user message, never a planner rewrite (CLAUDE.md).
Scores are sigmoid outputs in [0, 1], so ``RERANK_MIN_SCORE`` is an absolute threshold.
The model is loaded lazily on first use (or by ``warmup``) and runs in a worker thread so
it does not block the event loop; calls are serialized because one GPU model is shared.
"""

import asyncio
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from memory.config import Settings

log = logging.getLogger(__name__)


class CrossEncoderLike(Protocol):
    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Scored[T]:
    item: T
    score: float


def _load(settings: Settings) -> CrossEncoderLike:
    # Imported lazily: torch takes seconds to import, and tests and the API should not pay
    # that cost unless the reranker is actually used.
    import torch  # noqa: PLC0415
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    device = settings.reranker_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    log.info("loading reranker %s on %s", settings.reranker_model, device)
    model: CrossEncoderLike = CrossEncoder(
        settings.reranker_model,
        device=device,
        max_length=settings.reranker_max_length,
        model_kwargs=model_kwargs,
    )
    return model


class Reranker:
    """Cross-encoder scoring with two throughput measures (Phase 8 load test, 2026-10-25):

    - Pairs are scored in length order, so each GPU batch is padded only to similar lengths
      (2.4x faster on an RTX 3050 with identical scores); results return in input order.
    - Requests that arrive while the model is busy are scored together in the next call
      (about 15% more throughput under load). Nothing waits for others to arrive.
    """

    def __init__(self, settings: Settings, model: CrossEncoderLike | None = None) -> None:
        self._settings = settings
        self._model = model
        self._lock = threading.Lock()
        self._pending: list[tuple[list[tuple[str, str]], asyncio.Future[list[float]]]] = []
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.model_calls = 0

    def _get_model(self) -> CrossEncoderLike:
        with self._lock:
            if self._model is None:
                self._model = _load(self._settings)
            return self._model

    def warmup(self) -> None:
        self._predict([("warmup", "warmup")])

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._get_model()
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        with self._lock:
            self.model_calls += 1
            scores = model.predict(
                [pairs[i] for i in order],
                batch_size=self._settings.reranker_batch_size,
                show_progress_bar=False,
            )
        out = [0.0] * len(pairs)
        for rank, i in enumerate(order):
            out[i] = float(scores[rank])
        return out

    async def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:  # a new event loop (tests, restarts): start fresh
            self._loop, self._pending, self._worker = loop, [], None
        future: asyncio.Future[list[float]] = loop.create_future()
        self._pending.append((pairs, future))
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._drain())
        return await future

    async def _drain(self) -> None:
        """Score everything queued in one model call; repeat while more arrived meanwhile."""
        while self._pending:
            batch, self._pending = self._pending, []
            merged = [p for pairs, _ in batch for p in pairs]
            try:
                scores = await asyncio.to_thread(self._predict, merged)
            except Exception as e:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)
                continue
            start = 0
            for pairs, future in batch:
                if not future.done():
                    future.set_result(scores[start : start + len(pairs)])
                start += len(pairs)

    async def rerank[T](
        self, query: str, candidates: Sequence[T], text_of: Callable[[T], str]
    ) -> list[Scored[T]]:
        """Score each candidate against ``query``; best first. Ties keep input order."""
        if not candidates:
            return []
        pairs = [(query, text_of(c)) for c in candidates]
        scores = await self._score(pairs)
        ranked = sorted(enumerate(scores), key=lambda p: (-p[1], p[0]))
        return [Scored(item=candidates[i], score=s) for i, s in ranked]
