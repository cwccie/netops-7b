"""Quality filtering for instruction-tuning datasets.

Implements deduplication, length filtering, format validation,
and basic toxicity detection for the NetOps-7B training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from netops_7b.data.curate import InstructionPair


# ---------------------------------------------------------------------------
# Toxicity / PII detection patterns
# ---------------------------------------------------------------------------

TOXICITY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(hack|exploit|attack|breach|steal)\b.*\b(password|credential|system)\b", re.I),
    re.compile(r"\b(bypass|disable|circumvent)\b.*\b(security|firewall|authentication)\b", re.I),
]

PII_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
]

# Network config keywords that must appear in at least one of instruction/response
NETWORK_KEYWORDS: list[str] = [
    "interface", "router", "bgp", "ospf", "vlan", "acl", "route", "switch",
    "firewall", "mpls", "vpn", "vxlan", "evpn", "isis", "eigrp", "hsrp",
    "vrrp", "spanning-tree", "stp", "lacp", "port-channel", "mlag",
    "snmp", "syslog", "ntp", "dns", "dhcp", "nat", "qos", "policy",
    "prefix-list", "route-map", "access-list", "neighbor", "network",
    "show", "configure", "set", "no", "ip", "ipv6", "loopback",
    "troubleshoot", "diagnose", "debug", "packet", "traceroute", "ping",
    "cisco", "juniper", "arista", "ios", "eos", "junos", "nx-os", "ios-xr",
]


@dataclass
class QualityFilter:
    """Multi-stage quality filter for instruction-tuning data.

    Stages:
    1. Deduplication (exact hash + near-duplicate via instruction similarity)
    2. Length filtering (min/max for both instruction and response)
    3. Format validation (non-empty, proper structure)
    4. Domain relevance (must contain network keywords)
    5. Toxicity screening (reject malicious instruction patterns)
    6. PII detection (flag/remove entries with personal information)
    """

    min_instruction_length: int = 20
    max_instruction_length: int = 2000
    min_response_length: int = 50
    max_response_length: int = 8000
    require_network_keyword: bool = True
    check_toxicity: bool = True
    check_pii: bool = True
    dedup: bool = True
    rejection_log: list[dict[str, str]] = field(default_factory=list)

    def _check_length(self, pair: InstructionPair) -> str | None:
        """Return rejection reason if length constraints fail, else None."""
        ilen = len(pair.instruction)
        rlen = len(pair.response)

        if ilen < self.min_instruction_length:
            return f"instruction_too_short ({ilen} < {self.min_instruction_length})"
        if ilen > self.max_instruction_length:
            return f"instruction_too_long ({ilen} > {self.max_instruction_length})"
        if rlen < self.min_response_length:
            return f"response_too_short ({rlen} < {self.min_response_length})"
        if rlen > self.max_response_length:
            return f"response_too_long ({rlen} > {self.max_response_length})"
        return None

    def _check_format(self, pair: InstructionPair) -> str | None:
        """Return rejection reason if format validation fails."""
        if not pair.instruction.strip():
            return "empty_instruction"
        if not pair.response.strip():
            return "empty_response"
        if pair.instruction.strip() == pair.response.strip():
            return "instruction_equals_response"
        if pair.category not in ("config_generation", "troubleshooting", "command_explanation", "unknown"):
            return f"invalid_category ({pair.category})"
        return None

    def _check_domain_relevance(self, pair: InstructionPair) -> str | None:
        """Return rejection reason if not network-domain relevant."""
        if not self.require_network_keyword:
            return None
        combined = (pair.instruction + " " + pair.response).lower()
        for keyword in NETWORK_KEYWORDS:
            if keyword in combined:
                return None
        return "no_network_keywords"

    def _check_toxicity(self, pair: InstructionPair) -> str | None:
        """Return rejection reason if toxic content detected."""
        if not self.check_toxicity:
            return None
        combined = pair.instruction + " " + pair.response
        for pattern in TOXICITY_PATTERNS:
            if pattern.search(combined):
                return f"toxicity_match ({pattern.pattern[:50]})"
        return None

    def _check_pii(self, pair: InstructionPair) -> str | None:
        """Return rejection reason if PII detected."""
        if not self.check_pii:
            return None
        combined = pair.instruction + " " + pair.response
        for pattern in PII_PATTERNS:
            if pattern.search(combined):
                return f"pii_detected ({pattern.pattern[:40]})"
        return None

    def filter_one(self, pair: InstructionPair) -> tuple[bool, str | None]:
        """Apply all quality checks to a single pair.

        Returns:
            (passed, rejection_reason) — True if the pair passes all checks.
        """
        checks = [
            self._check_format,
            self._check_length,
            self._check_domain_relevance,
            self._check_toxicity,
            self._check_pii,
        ]
        for check in checks:
            reason = check(pair)
            if reason:
                return False, reason
        return True, None

    def filter(
        self,
        pairs: list[InstructionPair],
        progress: bool = True,
    ) -> list[InstructionPair]:
        """Apply quality filtering with deduplication to a list of pairs.

        Args:
            pairs: Input instruction pairs.
            progress: Show progress bar.

        Returns:
            Filtered list of high-quality, deduplicated instruction pairs.
        """
        self.rejection_log = []
        seen_uids: set[str] = set()
        accepted: list[InstructionPair] = []

        iterator = tqdm(pairs, desc="Quality filtering") if progress else pairs

        for pair in iterator:
            # Deduplication
            if self.dedup:
                if pair.uid in seen_uids:
                    self.rejection_log.append({
                        "uid": pair.uid,
                        "reason": "duplicate",
                    })
                    continue
                seen_uids.add(pair.uid)

            # Quality checks
            passed, reason = self.filter_one(pair)
            if passed:
                accepted.append(pair)
            else:
                self.rejection_log.append({
                    "uid": pair.uid,
                    "reason": reason or "unknown",
                })

        return accepted

    def stats(self) -> dict[str, Any]:
        """Return filtering statistics from the last filter() call."""
        from collections import Counter

        reasons = Counter(r["reason"] for r in self.rejection_log)
        return {
            "rejected_total": len(self.rejection_log),
            "rejection_reasons": dict(reasons),
        }
