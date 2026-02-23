"""Dataset curation pipeline for NetOps-7B.

Synthesizes instruction pairs from Cisco, Juniper, and Arista network
documentation across three task categories:
  - Configuration generation
  - Troubleshooting diagnosis
  - Command explanation

Target: 15,000–30,000 high-quality instruction/response pairs.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Template library — vendor-specific instruction templates
# ---------------------------------------------------------------------------

CISCO_CONFIG_TEMPLATES: list[dict[str, str]] = [
    {
        "instruction": "Generate a Cisco IOS configuration to set up OSPF area {area_id} on interface {interface} with network {network}/{mask}.",
        "response": (
            "router ospf {process_id}\n"
            " network {network} {wildcard} area {area_id}\n"
            "!\n"
            "interface {interface}\n"
            " ip ospf {process_id} area {area_id}\n"
            " ip ospf cost {cost}\n"
            " no shutdown"
        ),
    },
    {
        "instruction": "Create a Cisco IOS-XE configuration for a BGP neighbor {neighbor_ip} in AS {remote_as} with route-map {rm_name} applied inbound.",
        "response": (
            "router bgp {local_as}\n"
            " neighbor {neighbor_ip} remote-as {remote_as}\n"
            " neighbor {neighbor_ip} description {description}\n"
            " neighbor {neighbor_ip} update-source {source_intf}\n"
            " address-family ipv4 unicast\n"
            "  neighbor {neighbor_ip} activate\n"
            "  neighbor {neighbor_ip} route-map {rm_name} in\n"
            " exit-address-family"
        ),
    },
    {
        "instruction": "Write a Cisco NX-OS configuration to create VXLAN VNI {vni} mapped to VLAN {vlan_id} with multicast group {mcast_group}.",
        "response": (
            "vlan {vlan_id}\n"
            " vn-segment {vni}\n"
            "!\n"
            "interface nve1\n"
            " member vni {vni}\n"
            "  mcast-group {mcast_group}\n"
            " no shutdown"
        ),
    },
    {
        "instruction": "Configure a Cisco IOS access-list {acl_number} to permit traffic from {src_network}/{src_mask} to {dst_network}/{dst_mask} on port {port}.",
        "response": (
            "ip access-list extended {acl_name}\n"
            " permit tcp {src_network} {src_wildcard} {dst_network} {dst_wildcard} eq {port}\n"
            " deny ip any any log\n"
            "!\n"
            "interface {interface}\n"
            " ip access-group {acl_name} in"
        ),
    },
    {
        "instruction": "Generate a Cisco SD-WAN (Viptela) configuration for a site {site_id} with system-ip {system_ip} and two TLOC connections.",
        "response": (
            "system\n"
            " system-ip {system_ip}\n"
            " site-id {site_id}\n"
            " organization-name {org_name}\n"
            "!\n"
            "vpn 0\n"
            " interface ge0/0\n"
            "  ip address {wan1_ip}/30\n"
            "  tunnel-interface\n"
            "   encapsulation ipsec\n"
            "   color biz-internet\n"
            "  no shutdown\n"
            " interface ge0/1\n"
            "  ip address {wan2_ip}/30\n"
            "  tunnel-interface\n"
            "   encapsulation ipsec\n"
            "   color mpls\n"
            "  no shutdown"
        ),
    },
    {
        "instruction": "Write Cisco IOS-XR configuration for IS-IS routing on interface {interface} in area {area_tag} with metric {metric}.",
        "response": (
            "router isis {area_tag}\n"
            " is-type level-2-only\n"
            " net {net_address}\n"
            " address-family ipv4 unicast\n"
            "  metric-style wide\n"
            " !\n"
            " interface {interface}\n"
            "  point-to-point\n"
            "  address-family ipv4 unicast\n"
            "   metric {metric}\n"
            "  !"
        ),
    },
    {
        "instruction": "Configure HSRP version 2 group {group_id} on Cisco interface {interface} with virtual IP {vip} and priority {priority}.",
        "response": (
            "interface {interface}\n"
            " ip address {ip_address} {subnet_mask}\n"
            " standby version 2\n"
            " standby {group_id} ip {vip}\n"
            " standby {group_id} priority {priority}\n"
            " standby {group_id} preempt delay minimum 30\n"
            " standby {group_id} track {track_id} decrement {dec_value}"
        ),
    },
]

JUNIPER_CONFIG_TEMPLATES: list[dict[str, str]] = [
    {
        "instruction": "Generate a Junos configuration to create an OSPF area {area_id} on interface {interface} with hello-interval {hello_int} and dead-interval {dead_int}.",
        "response": (
            "set protocols ospf area {area_id} interface {interface} hello-interval {hello_int}\n"
            "set protocols ospf area {area_id} interface {interface} dead-interval {dead_int}\n"
            "set protocols ospf area {area_id} interface {interface} interface-type p2p"
        ),
    },
    {
        "instruction": "Write a Junos firewall filter {filter_name} to rate-limit traffic to {rate} and burst {burst_size} on interface {interface}.",
        "response": (
            "set firewall policer {policer_name} if-exceeding bandwidth-limit {rate} burst-size-limit {burst_size}\n"
            "set firewall policer {policer_name} then discard\n"
            "set firewall filter {filter_name} term rate-limit then policer {policer_name}\n"
            "set firewall filter {filter_name} term rate-limit then accept\n"
            "set interfaces {interface} unit 0 family inet filter input {filter_name}"
        ),
    },
    {
        "instruction": "Create a Junos EVPN-VXLAN configuration for VNI {vni} with route-distinguisher {rd} and route-target {rt}.",
        "response": (
            "set routing-instances {instance_name} instance-type virtual-switch\n"
            "set routing-instances {instance_name} vtep-source-interface lo0.0\n"
            "set routing-instances {instance_name} route-distinguisher {rd}\n"
            "set routing-instances {instance_name} vrf-target target:{rt}\n"
            "set routing-instances {instance_name} protocols evpn encapsulation vxlan\n"
            "set routing-instances {instance_name} vlans v{vni} vlan-id {vlan_id}\n"
            "set routing-instances {instance_name} vlans v{vni} vxlan vni {vni}"
        ),
    },
    {
        "instruction": "Configure Junos BGP group {group_name} with neighbor {neighbor_ip} in AS {remote_as} using authentication key.",
        "response": (
            "set protocols bgp group {group_name} type external\n"
            "set protocols bgp group {group_name} peer-as {remote_as}\n"
            "set protocols bgp group {group_name} neighbor {neighbor_ip}\n"
            "set protocols bgp group {group_name} authentication-key \"$9$encrypted_key\"\n"
            "set protocols bgp group {group_name} import {import_policy}\n"
            "set protocols bgp group {group_name} export {export_policy}\n"
            "set protocols bgp group {group_name} local-address {local_ip}"
        ),
    },
    {
        "instruction": "Write a Junos class-of-service configuration with scheduler-map {sched_map} assigning {percent}% bandwidth to queue {queue_num}.",
        "response": (
            "set class-of-service schedulers {sched_name} transmit-rate percent {percent}\n"
            "set class-of-service schedulers {sched_name} buffer-size percent {buf_percent}\n"
            "set class-of-service schedulers {sched_name} priority {priority}\n"
            "set class-of-service scheduler-maps {sched_map} forwarding-class {fc_name} scheduler {sched_name}\n"
            "set class-of-service interfaces {interface} scheduler-map {sched_map}"
        ),
    },
]

ARISTA_CONFIG_TEMPLATES: list[dict[str, str]] = [
    {
        "instruction": "Generate an Arista EOS configuration for BGP with AS {local_as}, neighbor {neighbor_ip} in AS {remote_as}, and BFD enabled.",
        "response": (
            "router bgp {local_as}\n"
            "   router-id {router_id}\n"
            "   neighbor {neighbor_ip} remote-as {remote_as}\n"
            "   neighbor {neighbor_ip} bfd\n"
            "   neighbor {neighbor_ip} send-community extended\n"
            "   address-family ipv4\n"
            "      neighbor {neighbor_ip} activate"
        ),
    },
    {
        "instruction": "Write an Arista EOS VXLAN configuration for VTEP interface with VLAN {vlan_id} mapped to VNI {vni} and source interface {source_intf}.",
        "response": (
            "interface Vxlan1\n"
            "   vxlan source-interface {source_intf}\n"
            "   vxlan udp-port 4789\n"
            "   vxlan vlan {vlan_id} vni {vni}\n"
            "   vxlan flood vtep {flood_vtep}\n"
            "!\n"
            "vlan {vlan_id}\n"
            "   name VXLAN_{vni}"
        ),
    },
    {
        "instruction": "Create an Arista EOS MLAG configuration with domain {domain_id}, peer-link {peer_link}, and peer address {peer_ip}.",
        "response": (
            "vlan {mlag_vlan}\n"
            "   name MLAG_PEER\n"
            "   trunk group MLAG_PEER\n"
            "!\n"
            "interface {peer_link}\n"
            "   switchport mode trunk\n"
            "   switchport trunk group MLAG_PEER\n"
            "   channel-group {po_number} mode active\n"
            "!\n"
            "interface Vlan{mlag_vlan}\n"
            "   ip address {local_ip}/30\n"
            "!\n"
            "mlag configuration\n"
            "   domain-id {domain_id}\n"
            "   local-interface Vlan{mlag_vlan}\n"
            "   peer-address {peer_ip}\n"
            "   peer-link Port-Channel{po_number}"
        ),
    },
    {
        "instruction": "Configure Arista EOS prefix-list {pl_name} and route-map {rm_name} to filter routes for {network}/{prefix_len}.",
        "response": (
            "ip prefix-list {pl_name} seq 10 permit {network}/{prefix_len}\n"
            "ip prefix-list {pl_name} seq 20 permit {default_route}\n"
            "!\n"
            "route-map {rm_name} permit 10\n"
            "   match ip address prefix-list {pl_name}\n"
            "   set local-preference {local_pref}\n"
            "!\n"
            "route-map {rm_name} deny 20"
        ),
    },
]

TROUBLESHOOTING_TEMPLATES: list[dict[str, str]] = [
    {
        "instruction": "A Cisco switch shows '%LINK-3-UPDOWN: Interface {interface}, changed state to down' every few minutes. The interface is connected to a {device_type}. What are the likely causes and how do you troubleshoot?",
        "response": (
            "This flapping behavior on {interface} connected to a {device_type} has several likely causes:\n\n"
            "**Layer 1 Issues:**\n"
            "1. Bad cable or patch cord — swap the cable and test\n"
            "2. Dirty or damaged SFP/optic — clean connectors; check `show interfaces {interface} transceiver` for Rx/Tx power levels\n"
            "3. Port or SFP hardware failure — try a different port/SFP\n\n"
            "**Layer 2/Config Issues:**\n"
            "4. Speed/duplex mismatch — verify with `show interfaces {interface} status` and set both sides explicitly\n"
            "5. Spanning-tree topology change — check `show spanning-tree detail` for TCN events\n"
            "6. EtherChannel misconfiguration — verify `show etherchannel summary`\n\n"
            "**Troubleshooting steps:**\n"
            "```\n"
            "show interfaces {interface}\n"
            "show interfaces {interface} counters errors\n"
            "show logging | include {interface}\n"
            "show cdp neighbors {interface} detail\n"
            "test cable-diagnostics tdr interface {interface}\n"
            "```\n\n"
            "Start with Layer 1, then work up. If errors increment on the interface counters (CRC, input errors, runts), the problem is almost certainly physical."
        ),
    },
    {
        "instruction": "BGP session between {router1} ({as1}) and {router2} ({as2}) is stuck in Active state. The routers are directly connected via {interface}. How do you diagnose and fix this?",
        "response": (
            "A BGP session stuck in Active state means the local router is attempting TCP connections to the peer but failing. Systematic diagnosis:\n\n"
            "**1. Verify IP connectivity:**\n"
            "```\n"
            "ping {peer_ip} source {local_ip}\n"
            "show ip route {peer_ip}\n"
            "```\n\n"
            "**2. Check BGP configuration on both sides:**\n"
            "```\n"
            "show running-config | section router bgp\n"
            "```\n"
            "Verify: neighbor IP, remote-as, update-source, ebgp-multihop (if needed)\n\n"
            "**3. Common causes:**\n"
            "- **Wrong neighbor IP** — must match the source IP the peer uses\n"
            "- **Missing update-source** — if using loopbacks, both sides need `neighbor X update-source Loopback0`\n"
            "- **ACL/firewall blocking TCP 179** — check `show ip access-lists` on interfaces in path\n"
            "- **TTL issue** — eBGP defaults to TTL=1; if not directly connected, add `ebgp-multihop`\n"
            "- **MD5 authentication mismatch** — check `show ip bgp neighbors {peer_ip}` for auth failures\n"
            "- **Passive peer** — one side configured as `neighbor X transport connection-mode passive`\n\n"
            "**4. Debug (use with caution):**\n"
            "```\n"
            "debug ip bgp {peer_ip} events\n"
            "debug ip tcp transactions\n"
            "```\n\n"
            "Most often it's an update-source mismatch or an ACL blocking TCP 179."
        ),
    },
    {
        "instruction": "A Juniper MX router shows 'RPD_OSPF_NBRDOWN' syslog for neighbor {neighbor_ip} on interface {interface}. The neighbor was previously Full. What is the diagnostic process?",
        "response": (
            "OSPF neighbor dropping from Full indicates a serious adjacency issue. Diagnosis:\n\n"
            "**1. Check current OSPF state:**\n"
            "```\n"
            "show ospf neighbor {neighbor_ip} detail\n"
            "show ospf interface {interface} detail\n"
            "show ospf statistics\n"
            "```\n\n"
            "**2. Review logs for timing:**\n"
            "```\n"
            "show log messages | match ospf | last 50\n"
            "show log messages | match \"{interface}\" | last 50\n"
            "```\n\n"
            "**3. Common causes and checks:**\n\n"
            "- **Dead timer expiry** (most common) — the neighbor didn't send hellos within the dead interval.\n"
            "  Check: CPU on remote router, interface errors, MTU\n"
            "- **MTU mismatch** — OSPF DBD exchange fails if MTU differs.\n"
            "  Check: `show interfaces {interface} | match mtu` on both sides\n"
            "- **Area mismatch** — both sides must be in the same area.\n"
            "- **Authentication failure** — check `show ospf interface {interface} detail` for auth errors\n"
            "- **Network type mismatch** — p2p vs broadcast must match\n"
            "- **Physical flap** — check `show interfaces {interface} extensive` for carrier transitions\n\n"
            "**4. Verify from the neighbor's perspective:**\n"
            "```\n"
            "show ospf neighbor detail\n"
            "show system processes extensive | match rpd\n"
            "```\n\n"
            "If the RPD process is at high CPU, OSPF hellos may be delayed, causing dead timer expiry."
        ),
    },
    {
        "instruction": "Arista switch {hostname} in an EVPN-VXLAN fabric cannot learn MAC addresses from remote VTEPs. Local MAC learning works. How do you troubleshoot?",
        "response": (
            "Remote MAC learning failure in EVPN-VXLAN while local learning works points to control-plane or underlay issues:\n\n"
            "**1. Verify VXLAN interface status:**\n"
            "```\n"
            "show interfaces Vxlan1\n"
            "show vxlan vtep\n"
            "show vxlan vni\n"
            "```\n\n"
            "**2. Check EVPN routes (control plane):**\n"
            "```\n"
            "show bgp evpn summary\n"
            "show bgp evpn route-type mac-ip\n"
            "show bgp evpn route-type imet\n"
            "```\n"
            "If Type-2 (MAC/IP) routes are missing from remote VTEPs, the problem is in BGP EVPN.\n\n"
            "**3. Verify underlay connectivity:**\n"
            "```\n"
            "ping {remote_vtep_ip} source {local_vtep_ip}\n"
            "show ip route {remote_vtep_ip}\n"
            "```\n"
            "VXLAN requires VTEP-to-VTEP reachability in the underlay.\n\n"
            "**4. Check VNI mapping:**\n"
            "```\n"
            "show vxlan vni {vni}\n"
            "```\n"
            "Ensure VLAN-to-VNI mapping is consistent across all VTEPs.\n\n"
            "**5. Common causes:**\n"
            "- BGP EVPN session not established with spine/RR\n"
            "- Route-target mismatch between VTEPs\n"
            "- VXLAN source-interface (loopback) not advertised in underlay IGP\n"
            "- MTU too small — VXLAN adds 50-byte overhead; underlay MTU should be >= 9214\n"
            "- Multicast group mismatch for flood-and-learn (if not using ingress replication)\n\n"
            "Start by verifying BGP EVPN session state and Type-2 route propagation."
        ),
    },
    {
        "instruction": "Users report intermittent packet loss on a network path traversing {router1} -> {router2} -> {router3}. All interfaces show up/up. How do you isolate the problem?",
        "response": (
            "Intermittent packet loss with interfaces up/up requires systematic isolation:\n\n"
            "**1. Establish baseline with extended pings:**\n"
            "```\n"
            "! From {router1}:\n"
            "ping {router2_ip} repeat 1000 size 1500 source {intf1}\n"
            "ping {router3_ip} repeat 1000 size 1500 source {intf1}\n"
            "```\n\n"
            "**2. Check each hop's interface health:**\n"
            "```\n"
            "show interfaces {interface} | include errors|drops|overruns|CRC\n"
            "show platform hardware fed active qos queue stats interface {interface}\n"
            "show policy-map interface {interface}\n"
            "```\n\n"
            "**3. Look for:**\n"
            "- **Output drops/queue drops** — indicates congestion. Check QoS policy and link utilization\n"
            "- **Input errors/CRC** — physical layer issue\n"
            "- **Buffer misses** — router running out of packet buffers\n"
            "- **Interface resets** — possible duplex mismatch or cable issue\n\n"
            "**4. Verify routing stability:**\n"
            "```\n"
            "show ip route {destination} | include via\n"
            "show ip cef {destination}\n"
            "show logging | include OSPF|BGP|EIGRP\n"
            "```\n"
            "Route flapping causes transient forwarding blackholes.\n\n"
            "**5. Check CPU and memory:**\n"
            "```\n"
            "show processes cpu history\n"
            "show memory statistics\n"
            "```\n"
            "High CPU can cause punt-path drops (control-plane packets dropped).\n\n"
            "**6. Capture if needed:**\n"
            "```\n"
            "monitor capture LOSS interface {interface} both\n"
            "monitor capture LOSS match ipv4 host {src} host {dst}\n"
            "monitor capture LOSS start\n"
            "```\n\n"
            "Most intermittent loss is caused by microbursts (brief congestion) — check output queue drops first."
        ),
    },
]

COMMAND_EXPLANATION_TEMPLATES: list[dict[str, str]] = [
    {
        "instruction": "Explain the Cisco IOS command: `show ip bgp summary`",
        "response": (
            "**`show ip bgp summary`** displays an overview of all BGP neighbor sessions.\n\n"
            "**Key fields:**\n"
            "- **Neighbor** — IP address of the BGP peer\n"
            "- **V** — BGP version (4)\n"
            "- **AS** — Remote autonomous system number\n"
            "- **MsgRcvd / MsgSent** — Total BGP messages received/sent (keepalives, updates, notifications)\n"
            "- **TblVer** — Current BGP table version; should match across all established peers\n"
            "- **InQ / OutQ** — Messages queued inbound/outbound; non-zero OutQ may indicate slow peer\n"
            "- **Up/Down** — How long the session has been established, or time since last state change\n"
            "- **State/PfxRcd** — If established, shows prefix count received; otherwise shows FSM state (Idle, Active, OpenSent, OpenConfirm)\n\n"
            "**Usage context:**\n"
            "This is the first command to run when troubleshooting BGP. Look for neighbors not in Established state, "
            "unexpected prefix counts, or sessions that keep resetting (short Up/Down timer).\n\n"
            "**Example output interpretation:**\n"
            "If State/PfxRcd shows 'Active', the TCP session isn't connecting. If it shows '0', the peer is established "
            "but sending no routes (check filters). If the number seems low, routes may be filtered or the peer may have issues."
        ),
    },
    {
        "instruction": "Explain the Junos command: `show route table inet.0 protocol bgp`",
        "response": (
            "**`show route table inet.0 protocol bgp`** displays all IPv4 unicast routes learned via BGP.\n\n"
            "**Breakdown:**\n"
            "- **`show route`** — Display the routing table\n"
            "- **`table inet.0`** — Filter to the IPv4 unicast routing table (Junos uses separate tables: "
            "inet.0 for IPv4 unicast, inet.2 for multicast RPF, inet6.0 for IPv6, etc.)\n"
            "- **`protocol bgp`** — Show only routes learned from the BGP protocol\n\n"
            "**Key output fields:**\n"
            "- **Destination prefix** — The network/mask\n"
            "- **Next-hop** — Where traffic is forwarded\n"
            "- **Preference** — Junos route preference (170 for eBGP, 170 for iBGP by default)\n"
            "- **AS path** — Sequence of ASNs the route traversed\n"
            "- **Communities** — BGP community values attached to the route\n"
            "- **Active route marker (*)** — Indicates this is the active route installed in the forwarding table\n\n"
            "**Useful variations:**\n"
            "- `show route table inet.0 protocol bgp active-path` — Only active (best) paths\n"
            "- `show route table inet.0 protocol bgp {prefix} detail` — Detailed view of a specific prefix\n"
            "- `show route table inet.0 protocol bgp terse` — Compact summary view"
        ),
    },
    {
        "instruction": "Explain the Arista EOS command: `show bgp evpn route-type mac-ip`",
        "response": (
            "**`show bgp evpn route-type mac-ip`** displays EVPN Type-2 routes, which carry MAC and optionally IP address information.\n\n"
            "**Context:** In EVPN-VXLAN fabrics, Type-2 routes are how switches advertise locally-learned MAC addresses "
            "(and their associated IPs) to remote VTEPs via BGP.\n\n"
            "**Key output fields:**\n"
            "- **Route Distinguisher (RD)** — Uniquely identifies the route's origin VTEP/VRF\n"
            "- **MAC address** — The host MAC being advertised\n"
            "- **IP address** — Associated host IP (if ARP/ND has been learned)\n"
            "- **VNI** — VXLAN Network Identifier for L2 and optionally L3 VNI\n"
            "- **Next-hop** — VTEP IP that originated the route\n"
            "- **Route-target** — Used to import the route into the correct VRF/VLAN on remote VTEPs\n"
            "- **ESI** — Ethernet Segment Identifier (for multi-homing)\n\n"
            "**When to use:**\n"
            "Use this to verify that MAC addresses from remote hosts are being advertised into the EVPN control plane. "
            "If a remote MAC is missing from this output, the originating VTEP isn't advertising it — check BGP EVPN session "
            "state and route-targets on the source."
        ),
    },
    {
        "instruction": "Explain the Cisco command: `show ip ospf database router`",
        "response": (
            "**`show ip ospf database router`** displays all Type-1 (Router) LSAs in the OSPF link-state database.\n\n"
            "**What Router LSAs contain:**\n"
            "Every OSPF router generates exactly one Router LSA per area it participates in. Each Router LSA describes:\n"
            "- **Links** — All interfaces the router has in that area\n"
            "- **Link types** — Point-to-point, transit network, stub network, virtual link\n"
            "- **Metrics** — Cost to reach each link\n"
            "- **Flags** — ABR (Area Border Router), ASBR (AS Boundary Router), Virtual link endpoint\n\n"
            "**Key fields in output:**\n"
            "- **Link State ID** — Router ID of the originating router\n"
            "- **Advertising Router** — Same as Link State ID for Type-1 LSAs\n"
            "- **LS Age** — Seconds since the LSA was originated (max 3600, then reflooded)\n"
            "- **Sequence Number** — Incremented each time the LSA is refreshed\n"
            "- **Checksum** — For integrity validation\n"
            "- **Link Count** — Number of links described in this LSA\n\n"
            "**Troubleshooting use:**\n"
            "Compare Router LSAs from different routers to verify the link-state database is consistent across the area. "
            "Missing LSAs or mismatched sequence numbers indicate OSPF flooding issues."
        ),
    },
    {
        "instruction": "Explain the Cisco IOS-XR command: `show mpls forwarding`",
        "response": (
            "**`show mpls forwarding`** displays the MPLS Label Forwarding Information Base (LFIB) — the table used to forward MPLS-labeled packets.\n\n"
            "**Key fields:**\n"
            "- **Local Label** — The label this router assigned and expects to receive on incoming packets\n"
            "- **Outgoing Label** — The label swapped onto the packet before forwarding (or 'Pop' for penultimate hop popping)\n"
            "- **Prefix or Tunnel ID** — The FEC (Forwarding Equivalence Class) this label maps to\n"
            "- **Outgoing Interface** — The interface the labeled packet is sent out\n"
            "- **Next Hop** — IP address of the downstream neighbor\n"
            "- **Bytes Switched** — Traffic counter for this label entry\n\n"
            "**Label operations:**\n"
            "- **Swap** — Replace incoming label with outgoing label (transit)\n"
            "- **Pop** — Remove the label (PHP — penultimate hop popping, or final hop)\n"
            "- **Push** — Add a label (ingress LER imposing labels)\n"
            "- **Aggregate** — Deliver to local IP forwarding (label maps to a local prefix)\n\n"
            "**When to use:**\n"
            "Verify that LSPs (Label Switched Paths) are correctly programmed. If traffic isn't following the expected path, "
            "check that label bindings are consistent with `show mpls ldp bindings` and that the LFIB matches the expected topology."
        ),
    },
]

# ---------------------------------------------------------------------------
# Parameter generators — produce realistic values for template slots
# ---------------------------------------------------------------------------

INTERFACES_CISCO = [
    "GigabitEthernet0/0", "GigabitEthernet0/1", "GigabitEthernet0/0/0",
    "TenGigabitEthernet1/0/1", "TenGigabitEthernet0/0/0",
    "Ethernet1/1", "Ethernet1/2", "Ethernet1/3",
    "Loopback0", "Loopback1", "Vlan100", "Vlan200",
    "Port-channel1", "Port-channel10",
]

INTERFACES_JUNIPER = [
    "ge-0/0/0", "ge-0/0/1", "ge-0/0/2", "ge-0/0/3",
    "xe-0/0/0", "xe-0/0/1", "et-0/0/0", "et-0/0/1",
    "ae0", "ae1", "lo0.0", "irb.100",
]

INTERFACES_ARISTA = [
    "Ethernet1", "Ethernet2", "Ethernet3", "Ethernet4",
    "Ethernet49/1", "Ethernet50/1",
    "Loopback0", "Loopback1",
    "Port-Channel1", "Port-Channel10",
    "Vlan100", "Vlan200",
]

NETWORKS = [
    ("10.0.0.0", "255.255.255.0", "0.0.0.255", "24"),
    ("10.1.0.0", "255.255.0.0", "0.0.255.255", "16"),
    ("172.16.0.0", "255.255.255.0", "0.0.0.255", "24"),
    ("172.16.1.0", "255.255.255.252", "0.0.0.3", "30"),
    ("192.168.1.0", "255.255.255.0", "0.0.0.255", "24"),
    ("192.168.100.0", "255.255.255.0", "0.0.0.255", "24"),
    ("10.100.0.0", "255.255.255.0", "0.0.0.255", "24"),
    ("10.200.0.0", "255.255.255.128", "0.0.0.127", "25"),
]


def _random_ip() -> str:
    """Generate a random private IP address."""
    ranges = [
        (10, random.randint(0, 255), random.randint(0, 255), random.randint(1, 254)),
        (172, random.randint(16, 31), random.randint(0, 255), random.randint(1, 254)),
        (192, 168, random.randint(0, 255), random.randint(1, 254)),
    ]
    octets = random.choice(ranges)
    return ".".join(str(o) for o in octets)


def _random_asn() -> int:
    """Generate a random AS number."""
    return random.choice(
        list(range(64512, 65535))  # private 2-byte
        + list(range(1, 1000))  # public range sample
        + [4200000000 + random.randint(0, 999999)]  # private 4-byte
    )


def _random_hostname() -> str:
    """Generate a realistic network device hostname."""
    roles = ["core", "dist", "access", "spine", "leaf", "border", "edge", "fw", "wan"]
    sites = ["dc1", "dc2", "hq", "br01", "br02", "pop1", "pop2", "colo1"]
    return f"{random.choice(roles)}-{random.choice(sites)}-{random.randint(1, 4):02d}"


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

@dataclass
class InstructionPair:
    """A single instruction/response training example."""

    instruction: str
    response: str
    category: str  # config_generation, troubleshooting, command_explanation
    vendor: str  # cisco, juniper, arista, multi-vendor
    complexity: str  # basic, intermediate, advanced
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        """Deterministic hash for deduplication."""
        content = f"{self.instruction}||{self.response}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "category": self.category,
            "vendor": self.vendor,
            "complexity": self.complexity,
            "uid": self.uid,
            **self.metadata,
        }


class ConfigGenerator:
    """Generates configuration instruction pairs by filling templates with realistic parameters."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _fill_cisco_params(self) -> dict[str, Any]:
        net = self.rng.choice(NETWORKS)
        return {
            "interface": self.rng.choice(INTERFACES_CISCO),
            "network": net[0],
            "mask": net[1],
            "wildcard": net[2],
            "prefix_len": net[3],
            "area_id": str(self.rng.choice([0, 1, 2, 10, 20, 51, 100])),
            "process_id": str(self.rng.randint(1, 100)),
            "cost": str(self.rng.choice([1, 10, 100, 1000])),
            "neighbor_ip": _random_ip(),
            "local_as": str(_random_asn()),
            "remote_as": str(_random_asn()),
            "rm_name": f"RM-{'IN' if self.rng.random() > 0.5 else 'OUT'}-{self.rng.randint(1, 10)}",
            "description": f"peer-{_random_hostname()}",
            "source_intf": self.rng.choice(["Loopback0", "Loopback1"]),
            "vni": str(self.rng.randint(10000, 99999)),
            "vlan_id": str(self.rng.randint(2, 4094)),
            "mcast_group": f"239.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}",
            "acl_number": str(self.rng.randint(100, 199)),
            "acl_name": f"ACL-{self.rng.choice(['PERMIT', 'DENY', 'MGMT', 'DATA'])}-{self.rng.randint(1, 20)}",
            "src_network": _random_ip().rsplit(".", 1)[0] + ".0",
            "src_mask": "255.255.255.0",
            "src_wildcard": "0.0.0.255",
            "dst_network": _random_ip().rsplit(".", 1)[0] + ".0",
            "dst_mask": "255.255.255.0",
            "dst_wildcard": "0.0.0.255",
            "port": str(self.rng.choice([22, 80, 443, 8080, 8443, 3389, 53, 123, 161, 179, 389, 636])),
            "site_id": str(self.rng.randint(100, 999)),
            "system_ip": _random_ip(),
            "org_name": "NetOps-Corp",
            "wan1_ip": _random_ip(),
            "wan2_ip": _random_ip(),
            "area_tag": f"CORE-{self.rng.randint(1, 5)}",
            "net_address": f"49.000{self.rng.randint(1, 9)}.{self.rng.randint(1000, 9999)}.{self.rng.randint(1000, 9999)}.{self.rng.randint(1000, 9999)}.00",
            "metric": str(self.rng.choice([10, 100, 1000, 10000])),
            "group_id": str(self.rng.randint(1, 255)),
            "vip": _random_ip(),
            "priority": str(self.rng.choice([110, 120, 150, 200])),
            "ip_address": _random_ip(),
            "subnet_mask": "255.255.255.0",
            "track_id": str(self.rng.randint(1, 50)),
            "dec_value": str(self.rng.choice([10, 20, 30, 50])),
        }

    def _fill_juniper_params(self) -> dict[str, Any]:
        return {
            "interface": self.rng.choice(INTERFACES_JUNIPER),
            "area_id": f"0.0.0.{self.rng.choice([0, 1, 2, 10])}",
            "hello_int": str(self.rng.choice([5, 10, 30])),
            "dead_int": str(self.rng.choice([20, 40, 120])),
            "filter_name": f"FILTER-{self.rng.choice(['RATE-LIMIT', 'PROTECT', 'COS'])}-{self.rng.randint(1, 10)}",
            "policer_name": f"POLICER-{self.rng.randint(1, 10)}m",
            "rate": f"{self.rng.choice([1, 5, 10, 50, 100])}m",
            "burst_size": f"{self.rng.choice([500, 1000, 5000])}k",
            "vni": str(self.rng.randint(10000, 99999)),
            "vlan_id": str(self.rng.randint(2, 4094)),
            "instance_name": f"VRF-{self.rng.choice(['PROD', 'DEV', 'MGMT', 'DMZ'])}",
            "rd": f"{_random_ip()}:{self.rng.randint(1, 65535)}",
            "rt": f"{_random_asn()}:{self.rng.randint(1, 65535)}",
            "group_name": f"EBGP-{self.rng.choice(['UPSTREAM', 'PEER', 'CUSTOMER'])}-{self.rng.randint(1, 5)}",
            "neighbor_ip": _random_ip(),
            "remote_as": str(_random_asn()),
            "import_policy": f"IMPORT-{self.rng.choice(['FULL', 'DEFAULT', 'PARTIAL'])}",
            "export_policy": f"EXPORT-{self.rng.choice(['FULL', 'DEFAULT', 'PARTIAL'])}",
            "local_ip": _random_ip(),
            "sched_map": f"SCHED-MAP-{self.rng.choice(['VOICE', 'DATA', 'VIDEO'])}",
            "sched_name": f"SCHED-{self.rng.choice(['EF', 'AF', 'BE'])}",
            "percent": str(self.rng.choice([5, 10, 20, 30, 50])),
            "buf_percent": str(self.rng.choice([5, 10, 20, 30])),
            "priority": self.rng.choice(["strict-high", "high", "medium-high", "low"]),
            "fc_name": self.rng.choice(["expedited-forwarding", "assured-forwarding", "best-effort", "network-control"]),
            "queue_num": str(self.rng.randint(0, 7)),
        }

    def _fill_arista_params(self) -> dict[str, Any]:
        return {
            "interface": self.rng.choice(INTERFACES_ARISTA),
            "local_as": str(_random_asn()),
            "remote_as": str(_random_asn()),
            "neighbor_ip": _random_ip(),
            "router_id": _random_ip(),
            "vlan_id": str(self.rng.randint(2, 4094)),
            "vni": str(self.rng.randint(10000, 99999)),
            "source_intf": "Loopback1",
            "flood_vtep": _random_ip(),
            "domain_id": f"MLAG-{self.rng.choice(['DC1', 'DC2', 'CAMPUS'])}",
            "peer_link": f"Ethernet{self.rng.choice([49, 50, 51, 52])}/1",
            "peer_ip": _random_ip(),
            "local_ip": _random_ip(),
            "mlag_vlan": str(self.rng.choice([4093, 4094])),
            "po_number": str(self.rng.randint(1, 100)),
            "pl_name": f"PL-{self.rng.choice(['CONNECTED', 'STATIC', 'DEFAULT', 'SUMMARY'])}",
            "rm_name": f"RM-{self.rng.choice(['BGP-IN', 'BGP-OUT', 'REDISTRIBUTE'])}",
            "network": _random_ip().rsplit(".", 1)[0] + ".0",
            "prefix_len": self.rng.choice(["24", "16", "25", "30"]),
            "default_route": "0.0.0.0/0",
            "local_pref": str(self.rng.choice([100, 150, 200, 250])),
        }

    def _fill_troubleshooting_params(self) -> dict[str, Any]:
        return {
            "interface": self.rng.choice(INTERFACES_CISCO + INTERFACES_JUNIPER + INTERFACES_ARISTA),
            "device_type": self.rng.choice(["server", "switch", "router", "firewall", "access point", "IP phone"]),
            "router1": _random_hostname(),
            "router2": _random_hostname(),
            "router3": _random_hostname(),
            "as1": str(_random_asn()),
            "as2": str(_random_asn()),
            "hostname": _random_hostname(),
            "neighbor_ip": _random_ip(),
            "peer_ip": _random_ip(),
            "local_ip": _random_ip(),
            "router2_ip": _random_ip(),
            "router3_ip": _random_ip(),
            "remote_vtep_ip": _random_ip(),
            "local_vtep_ip": _random_ip(),
            "vni": str(self.rng.randint(10000, 99999)),
            "intf1": self.rng.choice(INTERFACES_CISCO[:4]),
            "destination": _random_ip(),
            "src": _random_ip(),
            "dst": _random_ip(),
        }

    def generate_cisco(self, count: int) -> list[InstructionPair]:
        """Generate Cisco configuration instruction pairs."""
        results = []
        for _ in range(count):
            template = self.rng.choice(CISCO_CONFIG_TEMPLATES)
            params = self._fill_cisco_params()
            try:
                instruction = template["instruction"].format(**params)
                response = template["response"].format(**params)
                results.append(InstructionPair(
                    instruction=instruction,
                    response=response,
                    category="config_generation",
                    vendor="cisco",
                    complexity=self.rng.choice(["basic", "intermediate", "advanced"]),
                ))
            except KeyError:
                continue
        return results

    def generate_juniper(self, count: int) -> list[InstructionPair]:
        """Generate Juniper configuration instruction pairs."""
        results = []
        for _ in range(count):
            template = self.rng.choice(JUNIPER_CONFIG_TEMPLATES)
            params = self._fill_juniper_params()
            try:
                instruction = template["instruction"].format(**params)
                response = template["response"].format(**params)
                results.append(InstructionPair(
                    instruction=instruction,
                    response=response,
                    category="config_generation",
                    vendor="juniper",
                    complexity=self.rng.choice(["basic", "intermediate", "advanced"]),
                ))
            except KeyError:
                continue
        return results

    def generate_arista(self, count: int) -> list[InstructionPair]:
        """Generate Arista configuration instruction pairs."""
        results = []
        for _ in range(count):
            template = self.rng.choice(ARISTA_CONFIG_TEMPLATES)
            params = self._fill_arista_params()
            try:
                instruction = template["instruction"].format(**params)
                response = template["response"].format(**params)
                results.append(InstructionPair(
                    instruction=instruction,
                    response=response,
                    category="config_generation",
                    vendor="arista",
                    complexity=self.rng.choice(["basic", "intermediate", "advanced"]),
                ))
            except KeyError:
                continue
        return results

    def generate_troubleshooting(self, count: int) -> list[InstructionPair]:
        """Generate troubleshooting instruction pairs."""
        results = []
        for _ in range(count):
            template = self.rng.choice(TROUBLESHOOTING_TEMPLATES)
            params = self._fill_troubleshooting_params()
            try:
                instruction = template["instruction"].format(**params)
                response = template["response"].format(**params)
                results.append(InstructionPair(
                    instruction=instruction,
                    response=response,
                    category="troubleshooting",
                    vendor="multi-vendor",
                    complexity=self.rng.choice(["intermediate", "advanced"]),
                ))
            except KeyError:
                continue
        return results

    def generate_explanations(self, count: int) -> list[InstructionPair]:
        """Generate command explanation instruction pairs."""
        results = []
        available = list(COMMAND_EXPLANATION_TEMPLATES)
        for _ in range(count):
            if not available:
                available = list(COMMAND_EXPLANATION_TEMPLATES)
            template = self.rng.choice(available)
            results.append(InstructionPair(
                instruction=template["instruction"],
                response=template["response"],
                category="command_explanation",
                vendor="multi-vendor",
                complexity="intermediate",
            ))
        return results


class DatasetCurator:
    """Orchestrates the full dataset curation pipeline.

    Generates synthetic instruction pairs across vendors and task types,
    targeting 15K-30K examples with configurable distribution.
    """

    def __init__(
        self,
        target_size: int = 20000,
        seed: int = 42,
        distribution: dict[str, float] | None = None,
    ):
        self.target_size = target_size
        self.seed = seed
        self.generator = ConfigGenerator(seed=seed)
        self.distribution = distribution or {
            "cisco_config": 0.25,
            "juniper_config": 0.15,
            "arista_config": 0.15,
            "troubleshooting": 0.25,
            "explanation": 0.20,
        }

    def generate(self, progress: bool = True) -> list[InstructionPair]:
        """Generate the full dataset according to distribution."""
        pairs: list[InstructionPair] = []
        categories = {
            "cisco_config": self.generator.generate_cisco,
            "juniper_config": self.generator.generate_juniper,
            "arista_config": self.generator.generate_arista,
            "troubleshooting": self.generator.generate_troubleshooting,
            "explanation": self.generator.generate_explanations,
        }

        iterator = tqdm(self.distribution.items(), desc="Generating dataset") if progress else self.distribution.items()

        for category, fraction in iterator:
            count = int(self.target_size * fraction)
            gen_fn = categories[category]
            generated = gen_fn(count)
            pairs.extend(generated)

        return pairs

    def save(self, pairs: list[InstructionPair], output_path: str | Path, fmt: str = "jsonl") -> Path:
        """Save instruction pairs to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "jsonl":
            with open(output_path, "w") as f:
                for pair in pairs:
                    f.write(json.dumps(pair.to_dict()) + "\n")
        elif fmt == "json":
            with open(output_path, "w") as f:
                json.dump([p.to_dict() for p in pairs], f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return output_path

    def curate(self, output_path: str | Path, fmt: str = "jsonl", progress: bool = True) -> Path:
        """Full pipeline: generate, deduplicate, and save."""
        from netops_7b.data.quality import QualityFilter

        pairs = self.generate(progress=progress)
        qf = QualityFilter()
        pairs = qf.filter(pairs, progress=progress)
        return self.save(pairs, output_path, fmt=fmt)

    def stats(self, pairs: list[InstructionPair]) -> dict[str, Any]:
        """Return dataset statistics."""
        from collections import Counter

        cats = Counter(p.category for p in pairs)
        vendors = Counter(p.vendor for p in pairs)
        complexities = Counter(p.complexity for p in pairs)
        lengths = [len(p.instruction) + len(p.response) for p in pairs]

        return {
            "total": len(pairs),
            "categories": dict(cats),
            "vendors": dict(vendors),
            "complexities": dict(complexities),
            "avg_length_chars": sum(lengths) / len(lengths) if lengths else 0,
            "min_length_chars": min(lengths) if lengths else 0,
            "max_length_chars": max(lengths) if lengths else 0,
        }
