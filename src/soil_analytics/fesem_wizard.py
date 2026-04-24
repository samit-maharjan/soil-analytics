"""FESEM phase wizard: load YAML state machine, resolve a single phase id from a choice path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WizardOption:
    key: str
    label: str
    """Shorter line for 2×2 button layout; falls back to ``label`` in the UI if omitted."""
    short: str | None
    next: str | None
    result: str | None


@dataclass(frozen=True)
class WizardNode:
    id: str
    title: str
    prompt: str
    options: list[WizardOption]


@dataclass(frozen=True)
class WizardSpec:
    start: str
    nodes: dict[str, WizardNode]


def load_wizard(path: Path) -> WizardSpec:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    start = str(doc.get("start") or "")
    raw_nodes = doc.get("nodes") or {}
    out: dict[str, WizardNode] = {}
    for node_id, block in raw_nodes.items():
        if not isinstance(block, dict):
            continue
        opts: list[WizardOption] = []
        for o in block.get("options", []) or []:
            if not isinstance(o, dict):
                continue
            key = str(o.get("key") or "")
            label = str(o.get("label") or "")
            nxt = o.get("next")
            res = o.get("result")
            short = o.get("short")
            opts.append(
                WizardOption(
                    key=key,
                    label=label,
                    short=str(short).strip() if short else None,
                    next=str(nxt) if nxt else None,
                    result=str(res) if res else None,
                )
            )
        out[str(node_id)] = WizardNode(
            id=str(node_id),
            title=str(block.get("title") or ""),
            prompt=_prompt_text(block.get("prompt")),
            options=opts,
        )
    if start not in out:
        msg = f"start node {start!r} not found in nodes"
        raise ValueError(msg)
    return WizardSpec(start=start, nodes=out)


def _prompt_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if "\n" in s:
        return " ".join(line.strip() for line in s.splitlines() if line.strip())
    return s


def find_option(node: WizardNode, choice_key: str) -> WizardOption | None:
    for o in node.options:
        if o.key == choice_key:
            return o
    return None
