#!/usr/bin/env python3
"""
=============================================================================
TMF642 Multi-Vendor Alarm Normalisation Pipeline — Demo Runner
=============================================================================
Usage:
    python demo/run_demo.py                  # Run full cross-domain demo
    python demo/run_demo.py --scenario fiber # Run fiber-cut cascade scenario
    python demo/run_demo.py --vendor cisco   # Show only Cisco alarms
=============================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import List, Optional

from core.pipeline import NormalisationPipeline
from core.model import CanonicalAlarm, PerceivedSeverity
from data.test_samples import (
    ALL_SAMPLES, FIBER_CUT_CASCADE_SCENARIO,
    CISCO_SYSLOG_SAMPLES, NOKIA_RAN_SAMPLES, NOKIA_OPTICAL_SAMPLES,
    ERICSSON_ENM_SAMPLES, HUAWEI_IP_SAMPLES, SNMP_TRAP_SAMPLES,
    PROMETHEUS_SAMPLES, KUBERNETES_SAMPLES
)

# ANSI colour codes for terminal output
class C:
    RED    = "\033[91m"
    ORANGE = "\033[33m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    GREY   = "\033[90m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

SEV_COLOUR = {
    "critical":      C.RED,
    "major":         C.ORANGE,
    "minor":         C.YELLOW,
    "warning":       C.CYAN,
    "indeterminate": C.GREY,
    "cleared":       C.GREEN,
}

DOMAIN_COLOUR = {
    "ran":     "\033[94m",   # blue
    "optical": "\033[95m",   # magenta
    "ip":      "\033[36m",   # teal
    "compute": "\033[32m",   # green
    "core":    "\033[33m",   # amber
    "unknown": C.GREY,
}


def severity_badge(sev: str) -> str:
    c = SEV_COLOUR.get(sev.lower(), "")
    return f"{c}{C.BOLD}[{sev.upper():^13}]{C.RESET}"

def domain_badge(domain: str) -> str:
    c = DOMAIN_COLOUR.get(domain.lower(), "")
    return f"{c}{domain.upper():>8}{C.RESET}"


def print_alarm(alarm: CanonicalAlarm, index: int) -> None:
    sev   = alarm.perceived_severity.value
    dom   = alarm.x_domain.value
    ne    = alarm.alarmed_object.id if alarm.alarmed_object else "?"
    ne_name = alarm.alarmed_object.name if alarm.alarmed_object else ne
    ts    = alarm.alarm_raised_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n  {C.BOLD}#{index:02d}{C.RESET}  {severity_badge(sev)}  {domain_badge(dom)}  "
          f"{C.BOLD}{alarm.x_vendor or '?':10}{C.RESET}  {ts}")
    print(f"       NE         : {ne_name} ({ne})")
    print(f"       Alarm type : {alarm.alarm_type.value}")
    print(f"       Prob cause : {alarm.probable_cause}")
    print(f"       Spec prob  : {alarm.specific_problem or '—'}")
    if alarm.alarm_details:
        # Truncate details for readability
        detail = alarm.alarm_details[:110] + "…" if len(alarm.alarm_details) > 110 else alarm.alarm_details
        print(f"       Details    : {detail}")
    if alarm.service_affecting:
        print(f"       {C.RED}[SERVICE AFFECTING]{C.RESET}")
    if alarm.proposed_repair_actions:
        print(f"       Repair hint: {C.CYAN}{alarm.proposed_repair_actions[:90]}{C.RESET}")
    print(f"       TMF642 id  : {C.GREY}{alarm.id}{C.RESET}")


def print_tmf642_json(alarm: CanonicalAlarm) -> None:
    """Print the full TMF642-compliant JSON output."""
    print(json.dumps(alarm.to_dict(), indent=2, default=str))


def print_section(title: str) -> None:
    width = 76
    print(f"\n{'━' * width}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{'━' * width}")


def run_demo(samples: List, title: str, show_json: Optional[int] = None) -> None:
    """Run the pipeline on a list of samples and print results."""
    pipeline = NormalisationPipeline(enable_dedup=True)

    print_section(title)
    print(f"\n  {C.GREY}Processing {len(samples)} raw events…{C.RESET}")

    normalised = []
    for ev in samples:
        result = pipeline.process(
            raw_payload=ev["payload"],
            vendor=ev["vendor"],
            domain=ev["domain"],
            format=ev["format"],
        )
        if result:
            normalised.append((result, ev.get("description", "")))

    print(f"\n  {C.GREEN}✓ Normalised: {len(normalised)}{C.RESET}  |  "
          f"{C.GREY}Stats: {pipeline.stats}{C.RESET}\n")

    for i, (alarm, desc) in enumerate(normalised, start=1):
        if desc:
            print(f"  {C.GREY}╌╌ Source: {desc}{C.RESET}")
        print_alarm(alarm, i)

    # Optionally print one full TMF642 JSON
    if show_json is not None and 0 <= show_json < len(normalised):
        chosen = normalised[show_json][0]
        print_section(f"TMF642 JSON output — alarm #{show_json + 1}")
        print_tmf642_json(chosen)


def run_cascade_demo() -> None:
    """Special demo: show temporal cascade — optical → IP → RAN."""
    print_section("FIBER CUT CASCADE SCENARIO  (Optical → IP → RAN)")
    print(f"""
  {C.BOLD}Scenario:{C.RESET} A fiber cut on the Mumbai–Chennai span triggers a cascade:
    T+0s   Nokia 1830PSS detects OSNR degradation (early warning)
    T+10s  Nokia 1830PSS — Loss of Signal on OCH (fiber cut confirmed)
    T+10s  Nokia 1830PSS — EDFA amplifier fault on same span
    T+15s  Cisco PE router — GigabitEthernet interface down (IP circuit lost)
    T+20s  Cisco PE router — BGP peer down (cascades from interface)
    T+22s  Nokia gNB — 5G NR cell out of service (backhaul failure)
    T+25s  Nokia gNB — CPRI link failure (transport lost)

  {C.BOLD}Goal:{C.RESET} Normalise all events to TMF642 canonical form so downstream
  correlation engine can identify the optical LOS as root cause and
  suppress the 6 downstream alarms.
""")

    pipeline = NormalisationPipeline(enable_dedup=False)  # show all events in cascade
    for i, ev in enumerate(FIBER_CUT_CASCADE_SCENARIO, start=1):
        alarm = pipeline.process(
            raw_payload=ev["payload"],
            vendor=ev["vendor"],
            domain=ev["domain"],
            format=ev["format"],
        )
        if alarm:
            desc = ev.get("description", "")
            if desc:
                print(f"  {C.GREY}  {desc}{C.RESET}")
            print_alarm(alarm, i)

    # Show the first alarm as full JSON
    alarm = pipeline.process(
        raw_payload=NOKIA_OPTICAL_SAMPLES[0]["payload"],
        vendor="nokia", domain="optical", format="json_restconf",
    )
    if alarm:
        print_section("TMF642 JSON — optical root cause alarm (Nokia 1830PSS LOS)")
        print_tmf642_json(alarm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TMF642 Alarm Normalisation Pipeline Demo"
    )
    parser.add_argument(
        "--scenario", choices=["all", "fiber", "cisco", "nokia", "ericsson",
                               "huawei", "compute"],
        default="all", help="Which scenario to run"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Show full TMF642 JSON for the first result"
    )
    args = parser.parse_args()

    print(f"""
{C.BOLD}{'═' * 76}{C.RESET}
{C.BOLD}  TMF642 Multi-Vendor Alarm Normalisation Pipeline{C.RESET}
{C.BOLD}  Based on TM Forum TMF642 v4.0 / ITU-T X.733{C.RESET}
{C.BOLD}{'═' * 76}{C.RESET}
  Vendors:  Cisco · Nokia (NetAct + 1830PSS) · Ericsson ENM · Huawei iManager
  Domains:  RAN · IP/MPLS · Optical · Compute (Prometheus + Kubernetes)
  Formats:  Syslog · SNMP Trap · JSON REST · RESTCONF · Webhook · K8s Events
""")

    json_idx = 0 if args.json else None

    if args.scenario == "all":
        run_demo(CISCO_SYSLOG_SAMPLES,    "1 — CISCO IOS/XE  (IP domain, syslog format)", show_json=json_idx)
        run_demo(NOKIA_RAN_SAMPLES,       "2 — NOKIA NetAct  (RAN domain, RESTCONF JSON)")
        run_demo(NOKIA_OPTICAL_SAMPLES,   "3 — NOKIA 1830PSS (Optical domain, RESTCONF JSON)")
        run_demo(ERICSSON_ENM_SAMPLES,    "4 — ERICSSON ENM  (RAN domain, REST JSON)")
        run_demo(HUAWEI_IP_SAMPLES,       "5 — HUAWEI iManager (IP domain, REST JSON)")
        run_demo(SNMP_TRAP_SAMPLES,       "6 — Generic SNMP Traps (IF-MIB)")
        run_demo(PROMETHEUS_SAMPLES,      "7 — Prometheus AlertManager (Compute domain)")
        run_demo(KUBERNETES_SAMPLES,      "8 — Kubernetes Events (Compute domain)")
        run_cascade_demo()

    elif args.scenario == "fiber":
        run_cascade_demo()

    elif args.scenario == "cisco":
        run_demo(CISCO_SYSLOG_SAMPLES, "Cisco IOS/XE Syslog Alarms", show_json=json_idx)

    elif args.scenario == "nokia":
        run_demo(NOKIA_RAN_SAMPLES, "Nokia NetAct RAN Alarms", show_json=json_idx)
        run_demo(NOKIA_OPTICAL_SAMPLES, "Nokia 1830PSS Optical Alarms")

    elif args.scenario == "ericsson":
        run_demo(ERICSSON_ENM_SAMPLES, "Ericsson ENM Alarms", show_json=json_idx)

    elif args.scenario == "huawei":
        run_demo(HUAWEI_IP_SAMPLES, "Huawei iManager Alarms", show_json=json_idx)

    elif args.scenario == "compute":
        run_demo(PROMETHEUS_SAMPLES, "Prometheus AlertManager Alarms", show_json=json_idx)
        run_demo(KUBERNETES_SAMPLES, "Kubernetes Warning Events")

    print(f"\n{'━' * 76}\n  {C.GREEN}Demo complete.{C.RESET}\n")


if __name__ == "__main__":
    main()
