"""A terminal chatbot with long-term memory, built only on the public ``auramem`` API.

    docker compose up -d                       # Neo4j, Redis, Ollama
    uv run python examples/chatbot.py --user alice

Every session opens with the assistant picking up where the last one ended (from your profile
and the end of the previous conversation). Talk normally; memory is used and updated
automatically. Type /help for commands. Saying "forget that I ..." in plain language also
works: the bot shows what it found and asks first.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from auramem import AuraMem, Fact, Reply

DEFAULT_USER = "tejas-pokale"

# --- terminal helpers ---------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(text: str) -> str:
    return _c("2", text)


def bold(text: str) -> str:
    return _c("1", text)


def cyan(text: str) -> str:
    return _c("36", text)


def yellow(text: str) -> str:
    return _c("33", text)


def red(text: str) -> str:
    return _c("31", text)


def green(text: str) -> str:
    return _c("32", text)


HELP = """\
Commands
  /help                     this list
  /memory [all | words]     facts stored now (all: include replaced and hypothetical ones;
                            words: only facts containing them)
  /profile                  the always-on profile the assistant reads before every answer
  /recall <question>        what memory returns for a question, without answering it
  /forget <what>            show the matching memories, then delete them if you confirm
  /wipe                     permanently delete everything stored about this user
  /flush                    extract facts from this conversation now (otherwise automatic)
  /stats                    counts: messages, facts, entities, profile size, pending
  /mode auto|quick|deep     answering mode (auto: the planner decides per question)
  /deep <question>          answer one question with deep search
  /debug [on | off | context]  retrieval details after each answer (context: also the
                            memory block the model saw)
  /new                      start a new conversation (the current one is extracted first)
  /user <id>                switch to another user
  /export [file]            save everything stored about this user as JSON
  /quit                     exit (pending messages are extracted first)"""


def fact_line(f: Fact) -> str:
    when = f.start or f.mentioned_at
    tags = []
    if f.conflicts_with_ids:
        tags.append(yellow("conflicting"))
    elif f.attribute_key and not f.is_latest:
        tags.append(dim("replaced"))
    if f.is_hypothetical:
        tags.append(dim("hypothetical"))
    if f.attributed_to == "assistant":
        tags.append(dim("said by assistant"))
    suffix = f"  [{', '.join(tags)}]" if tags else ""
    return f"{dim(f'{when:%Y-%m-%d}')}  {f.text}{suffix}"


def debug_view(reply: Reply, seconds: float, *, context: bool) -> str:
    m = reply.memory
    plan = m.plan
    u = reply.usage
    lines = [
        dim("-- debug " + "-" * 50),
        dim(
            f"mode {reply.mode} | plan depth {plan.depth if plan else 'n/a'} "
            f"({m.plan_source or 'none'}) | {seconds:.1f} s | model {reply.model} "
            f"| prompt {reply.prompt_version}"
        ),
        dim(
            f"memory: {m.token_estimate} tokens, {len(m.fact_ids)} facts, "
            f"{len(m.turn_ids)} turns | top rerank "
            f"{'n/a' if m.top_rerank_score is None else f'{m.top_rerank_score:.2f}'} "
            f"| profile {reply.profile_tokens} tokens"
        ),
        dim(
            "timings: " + ", ".join(f"{k} {v:.0f} ms" for k, v in m.timings_ms.items())
            if m.timings_ms
            else "timings: n/a"
        ),
        dim(
            f"tokens: in {u.get('input_tokens', 0)} (cached {u.get('cache_read_tokens', 0)}) "
            f"out {u.get('output_tokens', 0)}"
        ),
    ]
    if plan is not None and plan.guidance:
        lines.append(dim("guidance: " + " | ".join(plan.guidance)))
    if reply.deep is not None:
        d = reply.deep
        lines.append(
            dim(f"deep: {d.steps} steps, {len(d.tool_calls)} searches, stopped: {d.stop_reason}")
        )
        lines += [dim(f"  search: {c}") for c in d.tool_calls]
    if context:
        lines += [dim("memory block:"), dim(m.text or "(nothing injected)")]
    lines.append(dim("-" * 59))
    return "\n".join(lines)


def ask_yes_no(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


# --- the chatbot -------------------------------------------------------------------------


class ChatBot:
    def __init__(self, mem: AuraMem, user: str, *, mode: str, debug: str) -> None:
        self.mem = mem
        self.user = user
        self.mode = mode
        self.debug = debug  # "off" | "on" | "context"
        self.session = self._new_session_id()

    def _new_session_id(self) -> str:
        return f"{self.user}/chat-{datetime.now():%Y%m%d-%H%M%S}"

    # each handler returns False to quit
    def handle(self, line: str) -> bool:
        if not line.startswith("/"):
            self.say(line, self.mode)
            return True
        command, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        handler = getattr(self, f"cmd_{command.lower()}", None)
        if handler is None:
            print(red(f"Unknown command /{command}. Type /help."))
            return True
        result: bool | None = handler(arg)
        return result is not False

    def open_conversation(self) -> None:
        """The assistant speaks first, continuing from the previous conversation."""
        greeting = self.mem.greet(self.user, session_id=self.session)
        print(f"{bold(cyan('assistant'))}: {greeting}")

    def say(self, message: str, mode: str) -> None:
        started = time.perf_counter()
        reply = self.mem.chat(self.user, message, session_id=self.session, mode=mode)  # type: ignore[arg-type]
        seconds = time.perf_counter() - started
        tag = {"deep": " (deep search)", "forget": " (memory)"}.get(reply.mode, "")
        print(f"{bold(cyan('assistant'))}{dim(tag)}: {reply.text}")
        if self.debug != "off":
            print(debug_view(reply, seconds, context=self.debug == "context"))

    # -- commands --

    def cmd_help(self, arg: str) -> None:
        print(HELP)

    def cmd_quit(self, arg: str) -> bool:
        return False

    cmd_exit = cmd_quit
    cmd_q = cmd_quit

    def cmd_memory(self, arg: str) -> None:
        history = arg == "all"
        facts = self.mem.facts(self.user, history=history)
        if arg and not history:
            words = arg.lower().split()
            facts = [f for f in facts if all(w in f.text.lower() for w in words)]
        if not facts:
            stats = self.mem.stats(self.user)
            hint = ""
            if stats["messages_awaiting_extraction"]:
                hint = (
                    f" ({stats['messages_awaiting_extraction']} messages are not extracted yet;"
                    " /flush extracts them now)"
                )
            print(dim("No facts stored" + (" matching that" if arg and not history else "") + hint))
            return
        label = "all facts" if history else "current facts"
        print(bold(f"{len(facts)} {label} for {self.user}:"))
        for n, f in enumerate(facts, start=1):
            print(f"{n:>4}. {fact_line(f)}")

    def cmd_profile(self, arg: str) -> None:
        profile = self.mem.profile(self.user)
        print(profile or dim("The profile is empty (it fills in as conversations are extracted)."))

    def cmd_recall(self, arg: str) -> None:
        if not arg:
            print(red("Usage: /recall <question>"))
            return
        recall = self.mem.recall(self.user, arg, session_id=self.session)
        print(dim(f"{recall.memory_tokens} tokens, {recall.mode} mode"))
        print(recall.context or dim("(nothing relevant in memory)"))

    def cmd_forget(self, arg: str) -> None:
        if not arg:
            print(red("Usage: /forget <what to forget>"))
            return
        preview = self.mem.forget(self.user, arg)
        print(preview.render())
        if preview.preview_id is None:
            return
        if not ask_yes_no(f"Permanently delete these {len(preview.items)} items?"):
            self.mem.cancel_forget(self.user, preview.preview_id)
            print(green("Kept everything."))
            return
        report = self.mem.confirm_forget(self.user, preview.preview_id)
        print(green(report.render()) if report else red("That preview expired; try again."))

    def cmd_wipe(self, arg: str) -> None:
        stats = self.mem.stats(self.user)
        print(
            red(
                f"This permanently deletes everything stored about '{self.user}': "
                f"{stats['messages']} messages, {stats['facts']} facts, {stats['entities']} "
                "entities, and the profile. It cannot be undone."
            )
        )
        try:
            typed = input(f"Type the user id ({self.user}) to confirm: ").strip()
        except EOFError:
            typed = ""
        if typed != self.user:
            print(green("Cancelled; nothing was deleted."))
            return
        counts = self.mem.delete_user(self.user)
        deleted = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        print(green(f"Deleted: {deleted or 'nothing was stored'}."))
        self.session = self._new_session_id()

    def cmd_flush(self, arg: str) -> None:
        before = self.mem.stats(self.user)
        print(dim("Extracting..."))
        self.mem.flush(self.user)
        after = self.mem.stats(self.user)
        print(
            green(
                f"Done: {after['facts'] - before['facts']} new facts, "
                f"profile {after['profile_tokens']} tokens."
            )
        )

    def cmd_stats(self, arg: str) -> None:
        for key, value in self.mem.stats(self.user).items():
            print(f"  {key.replace('_', ' '):<30} {value}")

    def cmd_mode(self, arg: str) -> None:
        if arg not in {"auto", "quick", "deep"}:
            print(red(f"Usage: /mode auto|quick|deep (now: {self.mode})"))
            return
        self.mode = arg
        print(green(f"Mode: {arg}"))

    def cmd_deep(self, arg: str) -> None:
        if not arg:
            print(red("Usage: /deep <question>"))
            return
        self.say(arg, "deep")

    def cmd_debug(self, arg: str) -> None:
        if arg in {"", "on"}:
            self.debug = "on" if arg == "on" or self.debug == "off" else "off"
        elif arg in {"off", "context"}:
            self.debug = arg
        else:
            print(red("Usage: /debug [on|off|context]"))
            return
        print(green(f"Debug: {self.debug}"))

    def cmd_new(self, arg: str) -> None:
        self.mem.flush(self.user)
        self.session = self._new_session_id()
        print(green("New conversation started."))
        self.open_conversation()

    def cmd_user(self, arg: str) -> None:
        if not arg:
            print(red(f"Usage: /user <id> (now: {self.user})"))
            return
        self.mem.flush(self.user)
        self.user = arg
        self.session = self._new_session_id()
        print(green(f"Now chatting as {self.user}."))
        self.open_conversation()

    def cmd_export(self, arg: str) -> None:
        path = Path(arg or f"{self.user}-memory-export.json")
        data = self.mem.export(self.user)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(green(f"Saved to {path.resolve()}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chatbot with long-term memory.")
    parser.add_argument("--user", default=DEFAULT_USER, help=f"default: {DEFAULT_USER}")
    parser.add_argument("--mode", choices=["auto", "quick", "deep"], default="auto")
    parser.add_argument("--debug", action="store_true", help="start with /debug on")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(dim("Connecting (Neo4j, embeddings, reranker)..."))
    try:
        mem = AuraMem().start()
    except Exception as e:
        print(red(f"Could not start memory: {e}"))
        print("Is `docker compose up -d` running, and is ANTHROPIC_API_KEY set in .env?")
        sys.exit(1)

    bot = ChatBot(mem, args.user, mode=args.mode, debug="on" if args.debug else "off")
    print(bold(f"Chatting as {bot.user}.") + dim(" Type /help for commands, /quit to exit."))
    try:
        bot.open_conversation()
    except Exception as e:  # the chat still works without an opening message
        print(red(f"(could not prepare an opening message: {type(e).__name__}: {e})"))
    try:
        while True:
            try:
                line = input(f"{bold(green('you'))}: ").strip()
            except EOFError:
                break
            if not line:
                continue
            try:
                if not bot.handle(line):
                    break
            except KeyboardInterrupt:
                print(dim("(interrupted)"))
            except Exception as e:  # keep the chat alive; show what went wrong
                print(red(f"Error: {type(e).__name__}: {e}"))
    except KeyboardInterrupt:
        print()
    finally:
        print(dim("Saving memories..."))
        try:
            mem.flush()
        finally:
            mem.close()


if __name__ == "__main__":
    main()
