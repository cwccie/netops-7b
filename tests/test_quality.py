"""Tests for quality filtering."""

from __future__ import annotations

import pytest

from netops_7b.data.curate import InstructionPair
from netops_7b.data.quality import QualityFilter


@pytest.fixture
def qf():
    return QualityFilter()


@pytest.fixture
def good_pair():
    return InstructionPair(
        instruction="Generate a Cisco IOS configuration for OSPF area 0 on interface GigabitEthernet0/0",
        response=(
            "router ospf 1\n"
            " network 10.0.0.0 0.0.0.255 area 0\n"
            "!\n"
            "interface GigabitEthernet0/0\n"
            " ip ospf 1 area 0\n"
            " no shutdown"
        ),
        category="config_generation",
        vendor="cisco",
        complexity="basic",
    )


class TestQualityFilterIndividual:
    def test_good_pair_passes(self, qf, good_pair):
        passed, reason = qf.filter_one(good_pair)
        assert passed is True
        assert reason is None

    def test_empty_instruction_fails(self, qf):
        pair = InstructionPair("", "valid response with network interface config",
                               "config_generation", "cisco", "basic")
        passed, reason = qf.filter_one(pair)
        assert passed is False
        assert "empty_instruction" in reason

    def test_empty_response_fails(self, qf):
        pair = InstructionPair("Configure OSPF on router interface", "",
                               "config_generation", "cisco", "basic")
        passed, reason = qf.filter_one(pair)
        assert passed is False
        assert "empty_response" in reason

    def test_short_instruction_fails(self, qf):
        pair = InstructionPair("Hi", "router ospf 1\n network 10.0.0.0 0.0.0.255 area 0 interface config",
                               "config_generation", "cisco", "basic")
        passed, reason = qf.filter_one(pair)
        assert passed is False
        assert "instruction_too_short" in reason

    def test_short_response_fails(self, qf):
        pair = InstructionPair("Configure OSPF area 0 on the router interface", "ospf 1",
                               "config_generation", "cisco", "basic")
        passed, reason = qf.filter_one(pair)
        assert passed is False
        assert "response_too_short" in reason

    def test_instruction_equals_response_fails(self, qf):
        text = "Configure OSPF on the router interface GigabitEthernet0/0 for the network"
        pair = InstructionPair(text, text, "config_generation", "cisco", "basic")
        passed, reason = qf.filter_one(pair)
        assert passed is False
        assert "instruction_equals_response" in reason

    def test_no_network_keywords_fails(self, qf):
        pair = InstructionPair(
            "Write a recipe for chocolate cake with frosting",
            "Mix flour, sugar, cocoa powder, eggs, and butter. Bake at 350F for 30 minutes. "
            "Let cool, then add frosting made from cream cheese and powdered sugar.",
            "unknown", "unknown", "basic",
        )
        passed, reason = qf.filter_one(pair)
        assert passed is False
        assert "no_network_keywords" in reason

    def test_domain_check_can_be_disabled(self):
        qf = QualityFilter(require_network_keyword=False)
        pair = InstructionPair(
            "Write a recipe for chocolate cake with detailed frosting instructions",
            "Mix flour, sugar, cocoa powder, eggs, and butter. Bake at 350F for 30 minutes. "
            "Let cool and then apply your favorite frosting. Serve and enjoy.",
            "unknown", "unknown", "basic",
        )
        passed, reason = qf.filter_one(pair)
        assert passed is True


class TestQualityFilterBatch:
    def test_dedup_removes_duplicates(self, qf, good_pair):
        pairs = [good_pair, good_pair, good_pair]
        result = qf.filter(pairs, progress=False)
        assert len(result) == 1

    def test_dedup_can_be_disabled(self, good_pair):
        qf = QualityFilter(dedup=False)
        pairs = [good_pair, good_pair, good_pair]
        result = qf.filter(pairs, progress=False)
        assert len(result) == 3

    def test_filter_mixed_quality(self, qf, good_pair):
        bad_pair = InstructionPair("", "resp", "config_generation", "cisco", "basic")
        pairs = [good_pair, bad_pair]
        result = qf.filter(pairs, progress=False)
        assert len(result) == 1
        assert result[0].uid == good_pair.uid

    def test_stats_after_filter(self, qf, good_pair):
        bad_pair = InstructionPair("", "resp", "config_generation", "cisco", "basic")
        pairs = [good_pair, bad_pair, good_pair]  # 1 good + 1 bad + 1 duplicate
        qf.filter(pairs, progress=False)
        stats = qf.stats()
        assert stats["rejected_total"] == 2
        assert "empty_instruction" in stats["rejection_reasons"]
        assert "duplicate" in stats["rejection_reasons"]

    def test_large_batch(self):
        """Test filtering works with larger batches."""
        from netops_7b.data.curate import ConfigGenerator

        gen = ConfigGenerator(seed=42)
        pairs = gen.generate_cisco(200)
        qf = QualityFilter()
        result = qf.filter(pairs, progress=False)
        assert len(result) > 0
        assert len(result) <= len(pairs)
