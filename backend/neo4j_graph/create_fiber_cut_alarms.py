"""
create_fiber_cut_alarms.py
==========================
Creates exactly 7 alarm nodes for the Mumbai-Chennai fiber cut scenario.
Wipes any existing Alarm nodes first for a clean start.

Usage:
    python create_fiber_cut_alarms.py --password YOUR_NEO4J_PASSWORD
    python create_fiber_cut_alarms.py --uri bolt://IP:7687 --password PW
"""

import argparse
from datetime import datetime, timedelta

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Run: pip install neo4j")
    raise


# ── The 7 fiber cut alarms ────────────────────────────────────────────────────
# Sequence follows the real propagation timeline:
#   t+00s  Optical OSNR warning   (Nokia 1830PSS, ROADM-MUM-01)
#   t+15s  Optical LOS critical   (Nokia 1830PSS, ROADM-MUM-01)  ← ROOT CAUSE
#   t+15s  Amplifier fault        (Nokia 1830PSS, AMP-MUM-CHN-01)
#   t+25s  IP link down           (Cisco EPN,     RTR-PE-MUM-01)
#   t+28s  BGP session down       (Cisco EPN,     RTR-PE-MUM-01)
#   t+30s  Cell OOS Mumbai A01    (Nokia NetAct,  gNB-MUM-SITE-A01)
#   t+32s  Cell OOS Mumbai B01    (Ericsson ENM,  gNB-MUM-SITE-B01)

BASE_TIME = datetime.utcnow()

def ts(offset_seconds):
    return (BASE_TIME + timedelta(seconds=offset_seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

ALARMS = [
    {
        "id":               "ALM-OPT-001",
        "domain":           "optical",
        "vendor":           "Nokia",
        "source":           "Nokia-1830PSS",
        "device_id":        "ROADM-MUM-01",
        "device_name":      "ROADM Mumbai 01",
        "alarm_type":       "qualityOfServiceAlarm",
        "severity":         "major",
        "probable_cause":   "signalQualityEvaluationFailure",
        "specific_problem": "OSNR_DEGRADATION",
        "description":      "OSNR on OCH-1-1-1-RX degrading. Value: 14.2 dB (threshold: 18.0 dB)",
        "is_root_cause":    False,
        "raised_time":      ts(0),
        "state":            "raised",
    },
    {
        "id":               "ALM-OPT-002",
        "domain":           "optical",
        "vendor":           "Nokia",
        "source":           "Nokia-1830PSS",
        "device_id":        "ROADM-MUM-01",
        "device_name":      "ROADM Mumbai 01",
        "alarm_type":       "communicationsAlarm",
        "severity":         "critical",
        "probable_cause":   "lossOfSignal",
        "specific_problem": "LOS",
        "description":      "Loss of Signal on OCH-1-1-1-TX. Rx Power: -40 dBm. Fiber span MUM-CHN-SPAN-1 cut.",
        "is_root_cause":    True,
        "raised_time":      ts(15),
        "state":            "raised",
    },
    {
        "id":               "ALM-OPT-003",
        "domain":           "optical",
        "vendor":           "Nokia",
        "source":           "Nokia-1830PSS",
        "device_id":        "AMP-MUM-CHN-01",
        "device_name":      "EDFA Amplifier Mum-Chn Span1",
        "alarm_type":       "equipmentAlarm",
        "severity":         "critical",
        "probable_cause":   "equipmentFailure",
        "specific_problem": "AMPLIFIER_FAULT",
        "description":      "EDFA output power below threshold: -35 dBm. Expected: +3 dBm.",
        "is_root_cause":    False,
        "raised_time":      ts(15),
        "state":            "raised",
    },
    {
        "id":               "ALM-IP-001",
        "domain":           "ip",
        "vendor":           "Cisco",
        "source":           "Cisco-EPN-Manager",
        "device_id":        "RTR-PE-MUM-01",
        "device_name":      "PE Router Mumbai 01",
        "alarm_type":       "communicationsAlarm",
        "severity":         "major",
        "probable_cause":   "communicationsSubsystemFailure",
        "specific_problem": "LINK_DOWN",
        "description":      "Interface TenGigE0/0/0/1 changed state to down. Underlaid optical circuit lost.",
        "is_root_cause":    False,
        "raised_time":      ts(25),
        "state":            "raised",
    },
    {
        "id":               "ALM-IP-002",
        "domain":           "ip",
        "vendor":           "Cisco",
        "source":           "Cisco-EPN-Manager",
        "device_id":        "RTR-PE-MUM-01",
        "device_name":      "PE Router Mumbai 01",
        "alarm_type":       "communicationsAlarm",
        "severity":         "major",
        "probable_cause":   "softwareProgramAbnormallyTerminated",
        "specific_problem": "BGP_SESSION_DOWN",
        "description":      "BGP neighbour 10.1.2.1 (RTR-PE-CHN-01) down. Hold timer expired.",
        "is_root_cause":    False,
        "raised_time":      ts(28),
        "state":            "raised",
    },
    {
        "id":               "ALM-RAN-001",
        "domain":           "ran",
        "vendor":           "Nokia",
        "source":           "Nokia-NetAct",
        "device_id":        "gNB-MUM-SITE-A01",
        "device_name":      "gNB Mumbai Alpha 01",
        "alarm_type":       "communicationsAlarm",
        "severity":         "critical",
        "probable_cause":   "communicationsSubsystemFailure",
        "specific_problem": "CELL_OUTAGE",
        "description":      "NR cells on gNB-MUM-SITE-A01 out of service. Backhaul transport failure on N2/N3 path.",
        "is_root_cause":    False,
        "raised_time":      ts(30),
        "state":            "raised",
    },
    {
        "id":               "ALM-RAN-002",
        "domain":           "ran",
        "vendor":           "Ericsson",
        "source":           "Ericsson-ENM",
        "device_id":        "gNB-MUM-SITE-B01",
        "device_name":      "gNB Mumbai Beta 01",
        "alarm_type":       "communicationsAlarm",
        "severity":         "critical",
        "probable_cause":   "transmissionError",
        "specific_problem": "BACKHAUL_TRANSPORT_FAILURE",
        "description":      "Backhaul N2 transport link failure on gNB-MUM-SITE-B01. Upstream fiber cut on Mumbai-Chennai span.",
        "is_root_cause":    False,
        "raised_time":      ts(32),
        "state":            "raised",
    },
]

# ── Propagation chain ─────────────────────────────────────────────────────────
# Each tuple is (parent_alarm_id, child_alarm_id)
PROPAGATION = [
    ("ALM-OPT-002", "ALM-OPT-001"),   # LOS root → OSNR warning (precursor)
    ("ALM-OPT-002", "ALM-OPT-003"),   # LOS root → Amplifier fault
    ("ALM-OPT-002", "ALM-IP-001"),    # LOS root → Link down
    ("ALM-IP-001",  "ALM-IP-002"),    # Link down → BGP session down
    ("ALM-IP-001",  "ALM-RAN-001"),   # Link down → Cell OOS Mumbai A01
    ("ALM-IP-001",  "ALM-RAN-002"),   # Link down → Backhaul N2 Transport Link Failure Mumbai B01
]


# ── Neo4j writer ──────────────────────────────────────────────────────────────

class AlarmWriter:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        print(f"Connected to Neo4j at {uri}")

    def wipe_alarms(self, session):
        result = session.run("MATCH (a:Alarm) DETACH DELETE a")
        print("  Wiped existing Alarm nodes.")

    def create_alarm(self, session, alarm):
        session.run("""
            MERGE (a:Alarm {id: $id})
            SET a.domain          = $domain,
                a.vendor          = $vendor,
                a.source          = $source,
                a.deviceId        = $device_id,
                a.deviceName      = $device_name,
                a.alarmType       = $alarm_type,
                a.severity        = $severity,
                a.probableCause   = $probable_cause,
                a.specificProblem = $specific_problem,
                a.description     = $description,
                a.isRootCause     = $is_root_cause,
                a.raisedTime      = $raised_time,
                a.state           = $state
        """, **alarm)

        # Link alarm to its device node in the topology
        result = session.run("""
            MATCH (a:Alarm {id: $alarm_id})
            MATCH (n) WHERE n.id = $device_id
            MERGE (a)-[:TRIGGERED_ON]->(n)
            RETURN n.id AS matched
        """, alarm_id=alarm["id"], device_id=alarm["device_id"])

        record = result.single()
        device_matched = record["matched"] if record else "NOT FOUND"
        root_tag = " [ROOT CAUSE]" if alarm["is_root_cause"] else ""
        print(f"  {alarm['id']:15s}  [{alarm['domain']:7s}]"
              f"  {alarm['specific_problem']:20s}"
              f"  → {device_matched}{root_tag}")

    def create_propagation(self, session, parent_id, child_id):
        session.run("""
            MATCH (parent:Alarm {id: $parent_id})
            MATCH (child:Alarm  {id: $child_id})
            MERGE (parent)-[:PROPAGATED_TO]->(child)
        """, parent_id=parent_id, child_id=child_id)

    def print_summary(self, session):
        print("\n" + "="*55)
        print("ALARM GRAPH SUMMARY")
        print("="*55)
        result = session.run("""
            MATCH (a:Alarm)
            OPTIONAL MATCH (a)-[:TRIGGERED_ON]->(n)
            RETURN a.id AS id,
                   a.domain AS domain,
                   a.severity AS severity,
                   a.specificProblem AS problem,
                   a.isRootCause AS root,
                   n.id AS device
            ORDER BY a.raisedTime
        """)
        for row in result:
            root_tag = " ← ROOT CAUSE" if row["root"] else ""
            print(f"  {row['id']:15s}  {row['domain']:7s}"
                  f"  {row['severity']:8s}"
                  f"  {row['problem']:22s}"
                  f"  {row['device']}{root_tag}")

        prop_result = session.run(
            "MATCH ()-[r:PROPAGATED_TO]->() RETURN count(r) AS cnt"
        )
        prop_count = prop_result.single()["cnt"]
        print(f"\n  PROPAGATED_TO edges: {prop_count}")
        print("="*55)

    def close(self):
        self.driver.close()


# ── Cascade query to print ────────────────────────────────────────────────────

CASCADE_QUERY = """
MATCH path = (root:Alarm {isRootCause:true})
  -[:PROPAGATED_TO*1..4]->(symptom:Alarm)
RETURN path
"""

DEMO_QUERIES = f"""
╔══════════════════════════════════════════════════════╗
║  Neo4j Browser queries for fiber cut demo           ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Stage 2 — Full topology with alarms                 ║
║  MATCH (n)-[r]->(m)                                  ║
║  WHERE NOT n:NetworkSlice AND NOT m:NetworkSlice      ║
║  RETURN n, r, m LIMIT 300                            ║
║                                                      ║
║  Stage 3 — Fiber cut propagation cascade             ║
║  {CASCADE_QUERY.strip():<50s}║
║                                                      ║
║  Alarm table view                                    ║
║  MATCH (a:Alarm)-[:TRIGGERED_ON]->(n)                ║
║  RETURN a.specificProblem, a.severity,               ║
║         a.vendor, n.name, a.isRootCause              ║
║  ORDER BY a.raisedTime                               ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create fiber cut alarms in Neo4j"
    )
    parser.add_argument("--uri",      default="bolt://localhost:7687")
    parser.add_argument("--user",     default="neo4j")
    parser.add_argument("--password", required=True)
    parser.add_argument("--no-wipe",  action="store_true",
                        help="Do not wipe existing alarms first")
    args = parser.parse_args()

    writer = AlarmWriter(args.uri, args.user, args.password)

    with writer.driver.session() as session:

        if not args.no_wipe:
            writer.wipe_alarms(session)

        print(f"\nCreating {len(ALARMS)} fiber cut alarms...")
        for alarm in ALARMS:
            writer.create_alarm(session, alarm)

        print(f"\nCreating {len(PROPAGATION)} propagation edges...")
        for parent_id, child_id in PROPAGATION:
            session.run("""
                MATCH (parent:Alarm {id: $parent_id})
                MATCH (child:Alarm  {id: $child_id})
                MERGE (parent)-[:PROPAGATED_TO]->(child)
            """, parent_id=parent_id, child_id=child_id)
            print(f"  {parent_id} → {child_id}")

        writer.print_summary(session)

    writer.close()
    print(DEMO_QUERIES)


if __name__ == "__main__":
    main()
