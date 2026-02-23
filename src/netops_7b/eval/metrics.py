"""Custom metrics for network domain evaluation.

Provides specialized scoring for network configuration quality,
troubleshooting accuracy, and command explanation completeness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Config syntax patterns by vendor
# ---------------------------------------------------------------------------

CISCO_IOS_PATTERNS = {
    "interface_decl": re.compile(r"^interface\s+\S+", re.MULTILINE),
    "router_decl": re.compile(r"^router\s+(ospf|bgp|eigrp|isis|rip)\s+\S*", re.MULTILINE),
    "ip_address": re.compile(r"ip\s+address\s+\d+\.\d+\.\d+\.\d+\s+\d+\.\d+\.\d+\.\d+"),
    "no_shutdown": re.compile(r"no\s+shutdown"),
    "access_list": re.compile(r"(ip\s+)?access-list\s+(standard|extended)\s+\S+", re.MULTILINE),
    "route_map": re.compile(r"route-map\s+\S+\s+(permit|deny)\s+\d+", re.MULTILINE),
    "neighbor": re.compile(r"neighbor\s+\d+\.\d+\.\d+\.\d+\s+remote-as\s+\d+"),
    "vlan": re.compile(r"^vlan\s+\d+", re.MULTILINE),
    "end_marker": re.compile(r"^!", re.MULTILINE),
}

JUNIPER_PATTERNS = {
    "set_command": re.compile(r"^set\s+\S+", re.MULTILINE),
    "protocols": re.compile(r"set\s+protocols\s+(ospf|bgp|isis|ldp|mpls|evpn)"),
    "interfaces": re.compile(r"set\s+interfaces\s+\S+"),
    "routing_instances": re.compile(r"set\s+routing-instances\s+\S+"),
    "firewall": re.compile(r"set\s+firewall\s+(filter|policer)\s+\S+"),
    "policy": re.compile(r"set\s+policy-options\s+\S+"),
    "class_of_service": re.compile(r"set\s+class-of-service\s+\S+"),
}

ARISTA_PATTERNS = {
    "interface_decl": re.compile(r"^interface\s+(Ethernet|Loopback|Vlan|Port-Channel|Vxlan)\S*", re.MULTILINE),
    "router_bgp": re.compile(r"^router\s+bgp\s+\d+", re.MULTILINE),
    "router_ospf": re.compile(r"^router\s+ospf\s+\d+", re.MULTILINE),
    "vxlan": re.compile(r"vxlan\s+(source-interface|vni|vlan)"),
    "mlag": re.compile(r"^mlag\s+configuration", re.MULTILINE),
    "prefix_list": re.compile(r"^ip\s+prefix-list\s+\S+", re.MULTILINE),
    "route_map": re.compile(r"^route-map\s+\S+\s+(permit|deny)\s+\d+", re.MULTILINE),
}


@dataclass
class ConfigSyntaxScore:
    """Result of config syntax validation."""

    valid_constructs: int
    total_patterns_checked: int
    vendor_detected: str
    structural_score: float  # 0.0-1.0
    completeness_score: float  # 0.0-1.0
    issues: list[str]

    @property
    def overall_score(self) -> float:
        return (self.structural_score * 0.6 + self.completeness_score * 0.4)


@dataclass
class TroubleshootingScore:
    """Result of troubleshooting response evaluation."""

    has_systematic_approach: bool
    has_verification_commands: bool
    has_root_cause_analysis: bool
    has_remediation_steps: bool
    layer_coverage: list[str]  # which OSI layers addressed
    score: float  # 0.0-1.0


@dataclass
class ExplanationScore:
    """Result of command explanation evaluation."""

    has_command_breakdown: bool
    has_field_descriptions: bool
    has_usage_context: bool
    has_examples: bool
    technical_depth: float  # 0.0-1.0
    score: float  # 0.0-1.0


class NetworkMetrics:
    """Custom metrics for evaluating network-domain LLM outputs."""

    def score_config_syntax(self, text: str, vendor: str = "auto") -> ConfigSyntaxScore:
        """Validate configuration syntax against vendor patterns.

        Uses regex-based AST-like parsing to check structural validity
        of generated network configurations.
        """
        if vendor == "auto":
            vendor = self._detect_vendor(text)

        patterns = self._get_vendor_patterns(vendor)
        valid = 0
        total = len(patterns)
        issues: list[str] = []

        for name, pattern in patterns.items():
            if pattern.search(text):
                valid += 1
            else:
                issues.append(f"Missing expected construct: {name}")

        # Structural checks
        structural_issues = self._check_structure(text, vendor)
        issues.extend(structural_issues)
        structural_penalty = len(structural_issues) * 0.1
        structural_score = max(0.0, 1.0 - structural_penalty)

        # Completeness: ratio of matched patterns
        completeness_score = valid / total if total > 0 else 0.0

        return ConfigSyntaxScore(
            valid_constructs=valid,
            total_patterns_checked=total,
            vendor_detected=vendor,
            structural_score=structural_score,
            completeness_score=completeness_score,
            issues=issues,
        )

    def score_troubleshooting(self, text: str) -> TroubleshootingScore:
        """Evaluate troubleshooting response quality."""
        text_lower = text.lower()

        # Check for systematic approach
        systematic_indicators = [
            "step", "first", "second", "then", "next",
            "verify", "check", "confirm", "diagnose",
        ]
        has_systematic = sum(1 for ind in systematic_indicators if ind in text_lower) >= 3

        # Check for verification commands
        command_patterns = [
            re.compile(r"(show|display|get|monitor)\s+\S+"),
            re.compile(r"(ping|traceroute|mtr|tcpdump)"),
            re.compile(r"debug\s+\S+"),
        ]
        has_commands = any(p.search(text) for p in command_patterns)

        # Check for root cause analysis
        root_cause_indicators = [
            "cause", "reason", "because", "due to", "root cause",
            "likely", "common", "typical", "usually",
        ]
        has_root_cause = sum(1 for ind in root_cause_indicators if ind in text_lower) >= 2

        # Check for remediation
        remediation_indicators = [
            "fix", "resolve", "solution", "remediat", "repair",
            "configure", "change", "modify", "update", "replace",
        ]
        has_remediation = sum(1 for ind in remediation_indicators if ind in text_lower) >= 2

        # Layer coverage
        layers = []
        layer_keywords = {
            "L1/Physical": ["cable", "physical", "optic", "sfp", "transceiver", "layer 1", "l1"],
            "L2/DataLink": ["mac", "vlan", "spanning-tree", "stp", "arp", "layer 2", "l2", "switch"],
            "L3/Network": ["ip", "route", "ospf", "bgp", "routing", "layer 3", "l3", "subnet"],
            "L4/Transport": ["tcp", "udp", "port", "session", "layer 4", "l4", "socket"],
            "L7/Application": ["dns", "http", "application", "layer 7", "l7", "service"],
        }
        for layer_name, keywords in layer_keywords.items():
            if any(kw in text_lower for kw in keywords):
                layers.append(layer_name)

        # Score calculation
        component_scores = [
            has_systematic * 0.25,
            has_commands * 0.25,
            has_root_cause * 0.20,
            has_remediation * 0.15,
            min(len(layers) / 3.0, 1.0) * 0.15,  # up to 3 layers for full score
        ]
        score = sum(component_scores)

        return TroubleshootingScore(
            has_systematic_approach=has_systematic,
            has_verification_commands=has_commands,
            has_root_cause_analysis=has_root_cause,
            has_remediation_steps=has_remediation,
            layer_coverage=layers,
            score=min(score, 1.0),
        )

    def score_explanation(self, text: str) -> ExplanationScore:
        """Evaluate command explanation quality."""
        text_lower = text.lower()

        # Check for command breakdown
        breakdown_indicators = ["syntax", "breakdown", "components", "parts", "format", "parameters"]
        has_breakdown = any(ind in text_lower for ind in breakdown_indicators) or "`" in text

        # Check for field descriptions
        field_indicators = ["field", "column", "output", "shows", "displays", "indicates"]
        has_fields = sum(1 for ind in field_indicators if ind in text_lower) >= 2

        # Check for usage context
        context_indicators = [
            "when to use", "useful for", "commonly used", "use this",
            "troubleshooting", "context", "scenario", "example",
        ]
        has_context = any(ind in text_lower for ind in context_indicators)

        # Check for examples
        has_examples = bool(re.search(r"```|example|sample|output:", text_lower))

        # Technical depth — based on specificity and detail
        depth_markers = [
            re.compile(r"\*\*.*?\*\*"),  # bold text (structured explanation)
            re.compile(r"```"),  # code blocks
            re.compile(r"\d+\.\d+\.\d+\.\d+"),  # IP addresses
            re.compile(r"\b(RFC|protocol|standard|specification)\b", re.I),
        ]
        depth_count = sum(1 for m in depth_markers if m.search(text))
        technical_depth = min(depth_count / 3.0, 1.0)

        # Word count factor (longer explanations tend to be more thorough)
        word_count = len(text.split())
        length_factor = min(word_count / 200.0, 1.0)

        # Score
        component_scores = [
            has_breakdown * 0.20,
            has_fields * 0.25,
            has_context * 0.20,
            has_examples * 0.15,
            technical_depth * 0.10,
            length_factor * 0.10,
        ]
        score = sum(component_scores)

        return ExplanationScore(
            has_command_breakdown=has_breakdown,
            has_field_descriptions=has_fields,
            has_usage_context=has_context,
            has_examples=has_examples,
            technical_depth=technical_depth,
            score=min(score, 1.0),
        )

    def _detect_vendor(self, text: str) -> str:
        """Auto-detect vendor from configuration text."""
        text_lower = text.lower()

        cisco_score = sum(1 for p in CISCO_IOS_PATTERNS.values() if p.search(text))
        juniper_score = sum(1 for p in JUNIPER_PATTERNS.values() if p.search(text))
        arista_score = sum(1 for p in ARISTA_PATTERNS.values() if p.search(text))

        # Additional heuristics
        if "set " in text and not text.strip().startswith("interface"):
            juniper_score += 3
        if "router bgp" in text_lower and ("Ethernet" in text or "Vxlan" in text):
            arista_score += 2
        if "!" in text and ("router" in text_lower or "interface" in text_lower):
            cisco_score += 2

        scores = {"cisco": cisco_score, "juniper": juniper_score, "arista": arista_score}
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _get_vendor_patterns(self, vendor: str) -> dict[str, re.Pattern]:
        """Get regex patterns for a vendor."""
        vendor_map = {
            "cisco": CISCO_IOS_PATTERNS,
            "juniper": JUNIPER_PATTERNS,
            "arista": ARISTA_PATTERNS,
        }
        return vendor_map.get(vendor, CISCO_IOS_PATTERNS)

    def _check_structure(self, text: str, vendor: str) -> list[str]:
        """Check structural integrity of configuration."""
        issues: list[str] = []

        if vendor == "cisco":
            # Check for unmatched interface blocks
            lines = text.strip().split("\n")
            in_block = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("interface ") or stripped.startswith("router "):
                    if in_block:
                        issues.append("Nested block detected (missing delimiter)")
                    in_block = True
                elif stripped == "!" or stripped == "exit":
                    in_block = False

        elif vendor == "juniper":
            # Junos set commands should all start with "set"
            for line in text.strip().split("\n"):
                stripped = line.strip()
                if stripped and not stripped.startswith("set ") and not stripped.startswith("#"):
                    issues.append(f"Non-set command in Junos config: {stripped[:50]}")

        return issues
