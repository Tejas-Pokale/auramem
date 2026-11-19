"""Versioned prompt templates.

Files are named ``<name>_<version>[.<part>].md`` (e.g. ``answer_v1.system.md``). Outputs
record ``Prompt.id`` so every result can be traced to the exact prompt text.
"""

from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    part: str | None
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"


@cache
def load_prompt(name: str, version: str, part: str | None = None) -> Prompt:
    filename = f"{name}_{version}" + (f".{part}" if part else "") + ".md"
    text = resources.files(__package__).joinpath(filename).read_text(encoding="utf-8")
    return Prompt(name=name, version=version, part=part, text=text)
