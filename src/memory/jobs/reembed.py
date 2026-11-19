"""Re-embedding job (PLAN.md section 5 "Background jobs", Phase 8).

When the embedding model or its prompts change, re-embed every Turn, Fact, and Entity:

1. Stage: embed each node's text with the new model into ``embedding_next`` (resumable:
   nodes already staged or already on the model are skipped). Search keeps using the old
   vectors meanwhile.
2. Swap: once nothing is left to stage, copy the staged vectors into ``embedding`` per label,
   in one transaction each, so an index never holds a mix of models.

The vector indexes are created for EMBEDDING_DIMS dimensions; a model with another size
needs a new index migration first, so the job refuses to run in that case.

    uv run python -m memory.jobs.reembed --model <new model> [--batch 64] [--stage-only]
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass, field

from memory.config import Settings, get_settings
from memory.embed import Embedder, EmbedderLike
from memory.store.neo4j import EMBEDDED_LABELS, Neo4jStore

log = logging.getLogger(__name__)


@dataclass
class ReembedReport:
    model: str
    staged: dict[str, int] = field(default_factory=dict)
    swapped: dict[str, int] = field(default_factory=dict)


async def reembed(
    store: Neo4jStore,
    embedder: EmbedderLike,
    settings: Settings,
    *,
    model: str,
    batch: int,
    swap: bool = True,
) -> ReembedReport:
    report = ReembedReport(model=model)
    for label in EMBEDDED_LABELS:
        staged = 0
        while pending := await store.reembed_pending(label, model, batch):
            texts = [text for _, text in pending]
            vectors = (await embedder.embed_documents(texts)).vectors
            if any(len(v) != settings.embedding_dims for v in vectors):
                raise ValueError(
                    f"{model} returns {len(vectors[0])} dimensions; the indexes have "
                    f"{settings.embedding_dims}. Add an index migration for the new size first."
                )
            await store.set_next_embeddings(
                label, model, [(i, v) for (i, _), v in zip(pending, vectors, strict=True)]
            )
            staged += len(pending)
            log.info("staged %d %s nodes with %s", staged, label, model)
        report.staged[label] = staged
    if not swap:
        return report
    for label in EMBEDDED_LABELS:
        if remaining := await store.reembed_remaining(label, model):
            raise RuntimeError(f"{remaining} {label} nodes are not staged yet; rerun the job")
        report.swapped[label] = await store.swap_embeddings(label, model)
    return report


async def main_async(model: str, batch: int, swap: bool) -> None:
    settings = get_settings().model_copy(update={"embedding_model": model})
    store = Neo4jStore.connect(settings)
    embedder = Embedder(settings)
    try:
        report = await reembed(store, embedder, settings, model=model, batch=batch, swap=swap)
        print(f"staged {report.staged}; swapped {report.swapped}")
    finally:
        await embedder.aclose()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=get_settings().embedding_model)
    parser.add_argument("--batch", type=int, default=get_settings().embedding_batch_size)
    parser.add_argument("--stage-only", action="store_true", help="stage vectors, don't swap")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main_async(args.model, args.batch, not args.stage_only))


if __name__ == "__main__":
    main()
