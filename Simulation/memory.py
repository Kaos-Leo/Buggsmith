"""Zel Beta - Persistent Causal Memory Store.

Stores and indexes distilled causal rules across generations, ensuring mutator
agents learn from past collisions and regressions instead of repeating them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class CausalMemoryStore:
    """Persistent storage for machine-distilled forensic rules and collision resolutions."""

    def __init__(self, storage_path: Path | str = "causal_rules.json"):
        self.storage_path = Path(storage_path).resolve()
        self.rules: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        """Load stored rules from JSON file if available."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.rules = data
            except Exception:
                self.rules = []

    def save(self) -> None:
        """Persist rules to storage_path."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(self.rules, indent=2), encoding="utf-8")
        except Exception:
            pass

    def add_rule(
        self,
        tier: str,
        sublayer: str,
        rule: str,
        commit: str,
        minimal_diff: str = "",
        artifact_symbol: str = "",
    ) -> None:
        """Record a newly synthesized causal rule."""
        entry = {
            "tier": tier,
            "sublayer": sublayer,
            "rule": rule,
            "commit": commit,
            "minimal_diff": minimal_diff,
            "artifact_symbol": artifact_symbol,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Deduplicate
        if not any(r.get("rule") == rule and r.get("tier") == tier for r in self.rules):
            self.rules.append(entry)
            self.save()

    def get_rules_for_agent(self, tier: str, sublayer: Optional[str] = None) -> List[str]:
        """Query top relevant causal rules for a specific tier and sublayer."""
        matching: List[str] = []
        for r in self.rules:
            if r.get("tier") == tier:
                if sublayer is None or r.get("sublayer") == sublayer or r.get("sublayer") == "General":
                    matching.append(r["rule"])
        return matching

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Return all active causal rules."""
        return list(self.rules)

    def clear(self) -> None:
        """Clear memory store."""
        self.rules = []
        if self.storage_path.exists():
            self.storage_path.unlink()
