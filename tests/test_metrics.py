"""Tests for custom network domain metrics."""

from __future__ import annotations

import pytest

from netops_7b.eval.metrics import NetworkMetrics


@pytest.fixture
def metrics():
    return NetworkMetrics()


class TestConfigSyntaxScoring:
    def test_cisco_config_scores_well(self, metrics):
        config = """
router ospf 1
 network 10.0.0.0 0.0.0.255 area 0
!
interface GigabitEthernet0/0
 ip address 10.0.0.1 255.255.255.0
 no shutdown
!
ip access-list extended BLOCK
 deny ip any any
!
route-map RM-OUT permit 10
"""
        score = metrics.score_config_syntax(config, vendor="cisco")
        assert score.vendor_detected == "cisco"
        assert score.valid_constructs > 3
        assert score.overall_score > 0.3

    def test_juniper_config_scores_well(self, metrics):
        config = """
set protocols ospf area 0.0.0.0 interface ge-0/0/0 hello-interval 10
set protocols bgp group EBGP peer-as 65001
set interfaces ge-0/0/0 unit 0 family inet address 10.0.0.1/24
set firewall filter PROTECT term allow then accept
set policy-options prefix-list MGMT 10.0.0.0/24
set routing-instances VRF-PROD instance-type vrf
set class-of-service schedulers EF transmit-rate percent 30
"""
        score = metrics.score_config_syntax(config, vendor="juniper")
        assert score.vendor_detected == "juniper"
        assert score.valid_constructs > 4
        assert score.overall_score > 0.3

    def test_arista_config_scores_well(self, metrics):
        config = """
interface Ethernet1
 no shutdown
!
interface Loopback0
 ip address 10.1.1.1/32
!
router bgp 65001
 neighbor 10.0.0.2 remote-as 65002
!
interface Vxlan1
 vxlan source-interface Loopback1
 vxlan vni 10100
!
mlag configuration
 domain-id DC1
!
ip prefix-list CONNECTED seq 10 permit 10.0.0.0/24
!
route-map BGP-OUT permit 10
 match ip address prefix-list CONNECTED
"""
        score = metrics.score_config_syntax(config, vendor="arista")
        assert score.vendor_detected == "arista"
        assert score.valid_constructs > 4

    def test_auto_detect_vendor_cisco(self, metrics):
        config = "router ospf 1\n network 10.0.0.0 0.0.0.255 area 0\n!\ninterface GigabitEthernet0/0"
        score = metrics.score_config_syntax(config, vendor="auto")
        assert score.vendor_detected == "cisco"

    def test_auto_detect_vendor_juniper(self, metrics):
        config = "set protocols ospf area 0 interface ge-0/0/0\nset protocols bgp group EBGP peer-as 65001\nset interfaces ge-0/0/0"
        score = metrics.score_config_syntax(config, vendor="auto")
        assert score.vendor_detected == "juniper"

    def test_empty_config_low_score(self, metrics):
        score = metrics.score_config_syntax("hello world", vendor="cisco")
        assert score.valid_constructs == 0
        assert score.completeness_score == 0.0


class TestTroubleshootingScoring:
    def test_good_troubleshooting_response(self, metrics):
        response = """
To troubleshoot this OSPF neighbor issue, follow these steps:

**Step 1: Verify interface status**
First, check the physical layer:
```
show interfaces GigabitEthernet0/0
show ip ospf interface GigabitEthernet0/0
```

**Step 2: Check OSPF configuration**
Next, verify the OSPF configuration matches on both sides:
- Area ID must match
- Hello/dead timers must match
- Authentication must match

The most common cause is an MTU mismatch between the two interfaces.
To fix this, configure matching MTU on both sides:
```
interface GigabitEthernet0/0
 ip mtu 1500
```

Then verify the neighbor comes up with `show ip ospf neighbor`.
"""
        score = metrics.score_troubleshooting(response)
        assert score.has_systematic_approach is True
        assert score.has_verification_commands is True
        assert score.has_root_cause_analysis is True
        assert score.score > 0.5

    def test_poor_troubleshooting_response(self, metrics):
        response = "Reboot the router."
        score = metrics.score_troubleshooting(response)
        assert score.score < 0.3

    def test_layer_coverage_detection(self, metrics):
        response = """
        Check the cable and SFP (physical layer).
        Verify VLAN and MAC address table (layer 2).
        Check IP routing and subnet (layer 3).
        Verify TCP connectivity on port 443 (layer 4).
        """
        score = metrics.score_troubleshooting(response)
        assert len(score.layer_coverage) >= 3


class TestExplanationScoring:
    def test_good_explanation(self, metrics):
        explanation = """
**`show ip bgp summary`** displays an overview of all BGP neighbor sessions.

**Key fields:**
- **Neighbor** — IP address of the BGP peer
- **AS** — Remote autonomous system number
- **State/PfxRcd** — Session state or prefix count

**When to use:**
This is the first command to run when troubleshooting BGP.

**Example output:**
```
Neighbor  V  AS  MsgRcvd  MsgSent  TblVer  InQ  OutQ  Up/Down  State/PfxRcd
10.0.0.1  4  65001  1234  5678  100  0  0  01:23:45  50
```
"""
        score = metrics.score_explanation(explanation)
        assert score.has_command_breakdown is True
        assert score.has_field_descriptions is True
        assert score.has_usage_context is True
        assert score.has_examples is True
        assert score.score > 0.5

    def test_poor_explanation(self, metrics):
        explanation = "It shows BGP."
        score = metrics.score_explanation(explanation)
        assert score.score < 0.3


class TestVendorDetection:
    def test_detect_cisco(self, metrics):
        vendor = metrics._detect_vendor("router ospf 1\n!\ninterface GigabitEthernet0/0")
        assert vendor == "cisco"

    def test_detect_juniper(self, metrics):
        vendor = metrics._detect_vendor("set protocols ospf area 0\nset interfaces ge-0/0/0\nset protocols bgp")
        assert vendor == "juniper"
