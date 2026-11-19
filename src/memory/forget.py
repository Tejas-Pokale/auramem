"""Forget (PLAN.md section 5 step 11, Phase 8): find, show, and only then hard-delete.

A forget request (``/forget`` or "forget that I ..." in chat) never deletes anything by itself.
It returns a preview of every matching memory, and deletion happens only when the user
confirms (user decision 2026-10-25). On confirmation, in this order:

1. Hard-delete the matched facts and turns with their embeddings, plus the source turns of
   the matched facts. Other facts that merely came from the same turn stay (a message about
   the move and the new job keeps the move when the job is forgotten). The chat turns of the
   request and the preview repeat the content, so they go with every fact derived from them.
2. Delete entities nothing refers to any more.
3. Rewrite the profile without the item, then purge every saved profile version (older
   versions may still hold it).
4. Clear and regenerate the summaries of the entities the deleted facts mentioned.
5. Record a content-free audit event (when, and how many of each).
"""

import logging
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from memory.config import Settings
from memory.embed import EmbedderLike
from memory.models import Fact, ForgetItem, ForgetPreview, ForgetReport, Turn
from memory.obs import prompt_label
from memory.observe import ReflectorLLM, SummaryLLM, estimate_tokens, summarize_entities
from memory.prompts import load_prompt
from memory.rerank import Reranker
from memory.retrieve import Candidate, candidate_key, rerank_text
from memory.retrieve.fusion import reciprocal_rank_fusion
from memory.retrieve.signals import fact_signals, turn_signals
from memory.store.neo4j import Neo4jStore, VersionConflictError

log = logging.getLogger(__name__)

# Commands only: the verb opens the message (after "please", "can you", ...) or follows "I want
# you to", so questions like "what did I forget to pack?" are not forget requests.
_FORGET = re.compile(
    r"^(?:(?:i (?:want|need|would like) you to|i'd like you to|you should|now)\s+)?"
    r"(?:(?:forget|erase)(?:\s+(?:that|about|what i (?:said|told you)(?: about)?|everything"
    r"(?: about)?|all about))*\s+(?P<what>.+)"
    r"|(?:delete|remove|wipe)\s+(?:what i (?:said|told you)|everything|all)"
    r"(?:\s+about)?\s+(?P<what2>.+?)(?:\s+from (?:your )?memory)?)$",
    re.IGNORECASE,
)
_PREFIX = re.compile(r"^(?:please|pls|can you|could you|would you|hey|ok(?:ay)?)[\s,]+", re.I)
CONFIRM = re.compile(
    r"^\s*(?:yes|yeah|yep|y|sure|ok(?:ay)?|confirm(?:ed)?|go ahead|do it|delete (?:it|them|all))"
    r"\b",
    re.IGNORECASE,
)
CANCEL = re.compile(
    r"^\s*(?:no|nope|n|cancel|stop|don'?t|do not|keep (?:it|them))\b", re.IGNORECASE
)


def forget_request(message: str) -> str | None:
    """What the user asks to forget, or None when the message is not a forget request."""
    text = _PREFIX.sub("", message.strip())
    m = _FORGET.match(text)
    if m is None:
        return None
    what = (m.group("what") or m.group("what2") or "").strip(" .!?\"'")
    return what or None


class Forgetter:
    def __init__(
        self,
        store: Neo4jStore,
        embedder: EmbedderLike,
        reranker: Reranker,
        settings: Settings,
        *,
        profile_llm: ReflectorLLM | None = None,
        summarizer: SummaryLLM | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings
        self._profile_llm = profile_llm
        self._summarizer = summarizer

    def _not_before(self, now: datetime) -> datetime:
        return now - timedelta(minutes=self._settings.forget_preview_ttl_minutes)

    async def preview(
        self,
        user_id: str,
        description: str,
        *,
        now: datetime,
        session_id: str | None = None,
        request_text: str = "",
    ) -> ForgetPreview:
        """Find every memory about ``description``; nothing is deleted."""
        facts, turns = await self._matches(user_id, description)
        # A matched fact takes its source turns with it (they hold the same content).
        turn_ids = {t.turn_id for t in turns} | {i for f in facts for i in f.source_turn_ids}
        extra_turns = await self._store.get_turns(
            user_id, sorted(turn_ids - {t.turn_id for t in turns})
        )
        all_turns = [*turns, *extra_turns]
        items = sorted(
            [_fact_item(f) for f in facts] + [_turn_item(t) for t in all_turns],
            key=lambda i: (i.date, i.kind, i.item_id),
        )
        if not items:
            return ForgetPreview(preview_id=None, description=description)
        preview = ForgetPreview(preview_id=uuid.uuid4().hex, description=description, items=items)
        assert preview.preview_id is not None
        await self._store.save_pending_forget(
            user_id, preview_id=preview.preview_id, session_id=session_id,
            description=description, fact_ids=preview.fact_ids, turn_ids=preview.turn_ids,
            request_text=request_text, now=now, not_before=self._not_before(now),
        )  # fmt: skip
        return preview

    async def _matches(self, user_id: str, description: str) -> tuple[list[Fact], list[Turn]]:
        s = self._settings
        vector = (await self._embedder.embed_query([description])).vectors[0]
        k = s.forget_signal_top_k
        signals: dict[str, Any] = {
            **await fact_signals(
                self._store, user_id, query_vector=vector, query_text=description, k=k
            ),
            **await turn_signals(
                self._store, user_id, query_vector=vector, query_text=description, k=k
            ),
        }
        ranked = {
            name: [(candidate_key(h.item), h.item) for h in hits] for name, hits in signals.items()
        }
        candidates: list[Candidate] = [f.item for f in reciprocal_rank_fusion(ranked, k=s.rrf_k)]
        scored = await self._reranker.rerank(description, candidates, rerank_text)
        kept = [x.item for x in scored if x.score >= s.forget_min_relevance]
        return (
            [c for c in kept if isinstance(c, Fact)],
            [c for c in kept if isinstance(c, Turn)],
        )

    async def cancel(self, user_id: str, preview_id: str) -> bool:
        return await self._store.delete_pending_forget(user_id, preview_id)

    async def confirm(self, user_id: str, preview_id: str, *, now: datetime) -> ForgetReport | None:
        """Delete what the preview showed. None when the preview is unknown or expired."""
        pending = await self._store.pending_forget(
            user_id, preview_id, not_before=self._not_before(now)
        )
        if pending is None:
            return None
        turn_ids = list(pending.get("turn_ids") or [])
        chat_turns: list[str] = []
        if pending.get("session_id"):  # the chat request and preview repeat the content
            chat_turns = await self._store.session_turns_with_text(
                user_id, pending["session_id"],
                [pending.get("request_text") or "", pending.get("preview_text") or ""],
            )  # fmt: skip
        turn_ids += chat_turns
        derived = await self._store.facts_derived_from(user_id, chat_turns)
        fact_ids = list(
            dict.fromkeys([*(pending.get("fact_ids") or []), *(f.fact_id for f in derived)])
        )
        facts = await self._store.get_facts(user_id, fact_ids)
        turns = await self._store.get_turns(user_id, turn_ids)
        entity_ids = sorted(
            {e for f in facts for e in f.mentioned_entity_ids}
            | {f.subject_entity_id for f in facts if f.subject_entity_id}
        )
        counts = await self._store.delete_by_ids(user_id, fact_ids=fact_ids, turn_ids=turn_ids)
        orphans = await self._store.delete_orphan_entities(user_id, entity_ids)
        rewritten = await self._rewrite_profile(
            user_id, str(pending.get("description") or ""), [f.text for f in facts], turns
        )
        await self._store.purge_observation_log_versions(user_id)
        refreshed = await self._refresh_summaries(user_id, entity_ids)
        await self._store.record_deletion(
            user_id, kind="forget", facts=counts.facts, turns=counts.turns,
            entities=orphans, now=now,
        )  # fmt: skip
        await self._store.delete_pending_forget(user_id, preview_id)
        return ForgetReport(
            facts=counts.facts,
            turns=counts.turns,
            entities=orphans,
            profile_rewritten=rewritten,
            summaries_refreshed=refreshed,
        )

    async def _rewrite_profile(
        self, user_id: str, description: str, fact_texts: Sequence[str], turns: Sequence[Turn]
    ) -> bool:
        current = await self._store.get_observation_log(user_id)
        if current is None or self._profile_llm is None:
            return False
        version = self._settings.forget_prompt_version
        deleted = "\n".join(f"- {t}" for t in [*fact_texts, *(t.text for t in turns)])
        messages: list[BaseMessage] = [
            SystemMessage(load_prompt("forget", version, "system").text),
            HumanMessage(
                load_prompt("forget", version, "user").text.format(
                    description=description, deleted=deleted or "(none)", log=current.text
                )
            ),
        ]
        with prompt_label(f"forget_{version}"):
            output = await self._profile_llm(messages)
        text = "\n".join(line.rstrip() for line in output.text.splitlines() if line.strip())
        try:
            await self._store.replace_observation_log(
                user_id,
                text,
                estimate_tokens(text, self._settings),
                expected_version=current.version,
                reason=f"forget_{version}",
            )
        except VersionConflictError:
            log.warning("profile changed during forget for %s; rewriting again", user_id)
            return await self._rewrite_profile(user_id, description, fact_texts, turns)
        return True

    async def _refresh_summaries(self, user_id: str, entity_ids: Sequence[str]) -> int:
        """Old summaries may repeat forgotten facts: clear them, then rebuild from what's left."""
        remaining = [e.entity_id for e in await self._store.get_entities(user_id, entity_ids)]
        await self._store.clear_entity_summaries(user_id, remaining)
        if self._summarizer is None or not remaining:
            return 0
        return await summarize_entities(
            self._store, self._summarizer, self._settings, user_id, remaining
        )


def _fact_item(f: Fact) -> ForgetItem:
    return ForgetItem(kind="fact", item_id=f.fact_id, text=f.text, date=f.start or f.mentioned_at)


def _turn_item(t: Turn) -> ForgetItem:
    return ForgetItem(kind="turn", item_id=t.turn_id, text=t.text, date=t.created_at, role=t.role)
