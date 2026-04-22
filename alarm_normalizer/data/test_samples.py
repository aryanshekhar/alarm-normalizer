"""
Real-world representative alarm and event samples for all supported
vendors and domains.  Used by the demo and test suite.
"""
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# CISCO IOS/XE SYSLOG (raw string format, as received over UDP 514)
# ─────────────────────────────────────────────────────────────────────────────
CISCO_SYSLOG_SAMPLES: List[Dict[str, Any]] = [
    {
        "payload": {
            "timestamp": "2024-01-15T14:23:01Z",
            "hostname": "router-pe1",
            "facility": "LINK",
            "severity": "3",
            "mnemonic": "UPDOWN",
            "message": "Interface GigabitEthernet0/0/1, changed state to down"
        },
        "vendor": "cisco", "domain": "ip", "format": "syslog",
        "description": "Cisco PE router — interface GigabitEthernet0/0/1 link down (peer-side BGP peer)"
    },
    {
        "payload": {
            "timestamp": "2024-01-15T14:23:45Z",
            "hostname": "router-pe1",
            "facility": "BGP",
            "severity": "5",
            "mnemonic": "ADJCHG",
            "message": "neighbor 10.0.0.2 Down Interface flap"
        },
        "vendor": "cisco", "domain": "ip", "format": "syslog",
        "description": "BGP neighbour down — cascades from the interface failure above"
    },
    {
        "payload": {
            "timestamp": "2024-01-15T16:10:00Z",
            "hostname": "agg-sw-dc1",
            "facility": "HARDWARE",
            "severity": "2",
            "mnemonic": "FAN_FAILURE",
            "message": "Fan 2 in slot 1 has failed. Chassis temperature may rise."
        },
        "vendor": "cisco", "domain": "ip", "format": "syslog",
        "description": "Cisco Catalyst — chassis fan failure (environmental alarm)"
    },
    {
        "payload": {
            "timestamp": "2024-01-15T18:00:01Z",
            "hostname": "agg-sw-dc1",
            "facility": "HARDWARE",
            "severity": "3",
            "mnemonic": "CPU_OVERLOAD",
            "message": "CPU utilization for five seconds: 94%/58%; one minute: 89%; five minutes: 76%"
        },
        "vendor": "cisco", "domain": "ip", "format": "syslog",
        "description": "High CPU utilisation — possible routing protocol storm or DDoS"
    },
    {
        "payload": {
            "timestamp": "2024-01-15T14:25:00Z",
            "hostname": "router-pe1",
            "facility": "LINK",
            "severity": "5",
            "mnemonic": "UPDOWN",
            "message": "Interface GigabitEthernet0/0/1, changed state to up"
        },
        "vendor": "cisco", "domain": "ip", "format": "syslog",
        "description": "Cisco PE router — interface restored (clearing event)"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# NOKIA NETACT — RAN (JSON RESTCONF northbound)
# ─────────────────────────────────────────────────────────────────────────────
NOKIA_RAN_SAMPLES: List[Dict[str, Any]] = [
    {
        "payload": {
            "notificationId": "20240115-001234",
            "eventTime": "2024-01-15T14:22:45Z",
            "alarmType": "EQUIPMENT_ALARM",
            "probableCause": "RRH_FAULT",
            "perceivedSeverity": "MAJOR",
            "specificProblem": "Radio Unit Hardware Fault",
            "managedObjectClass": "BTS",
            "managedObjectInstance": "PLMN-PLMN/BSC-1/BTS-101",
            "additionalText": "RRH unit temperature exceeded threshold 85°C. Sector 1 impacted.",
            "alarmId": "BTS-101-RRH-FAULT-20240115",
            "neId": "gNB-SITE-ALPHA-01",
            "neName": "SITE-ALPHA-gNB1",
            "vendor": "Nokia",
            "domain": "RAN",
            "clearingTime": None,
            "serviceAffecting": True,
            "proposedRepairAction": "Replace RRH unit or check cooling; verify CPRI link integrity"
        },
        "vendor": "nokia", "domain": "ran", "format": "json_restconf",
        "description": "Nokia gNB — RRH hardware fault, sector 1 degraded"
    },
    {
        "payload": {
            "notificationId": "20240115-001235",
            "eventTime": "2024-01-15T14:23:10Z",
            "alarmType": "COMMUNICATIONS_ALARM",
            "probableCause": "CPRI_FAILURE",
            "perceivedSeverity": "CRITICAL",
            "specificProblem": "CPRI Link Loss of Signal",
            "managedObjectClass": "RRH",
            "managedObjectInstance": "PLMN-PLMN/BSC-1/BTS-101/RRH-1",
            "additionalText": "CPRI link between BBU and RRH-1 has lost signal. All cells on this RRH are out of service.",
            "alarmId": "BTS-101-CPRI-LOS-20240115",
            "neId": "gNB-SITE-ALPHA-01",
            "neName": "SITE-ALPHA-gNB1",
            "clearingTime": None,
            "serviceAffecting": True,
        },
        "vendor": "nokia", "domain": "ran", "format": "json_restconf",
        "description": "Nokia gNB — CPRI link failure between BBU and RRH (all cells on this RRH OOS)"
    },
    {
        "payload": {
            "notificationId": "20240115-001290",
            "eventTime": "2024-01-15T14:22:30Z",
            "alarmType": "COMMUNICATIONS_ALARM",
            "probableCause": "CELL_OUTAGE",
            "perceivedSeverity": "CRITICAL",
            "specificProblem": "NR Cell Out of Service",
            "managedObjectClass": "NRCellDU",
            "managedObjectInstance": "PLMN-PLMN/gNB-101/NRCellDU-ALPHA-1",
            "additionalText": "5G NR Cell ALPHA-1 is out of service due to transport failure.",
            "alarmId": "gNB-101-CELL-OOS-ALPHA-1",
            "neId": "gNB-SITE-ALPHA-01",
            "neName": "SITE-ALPHA-gNB1",
            "clearingTime": None,
            "serviceAffecting": True,
        },
        "vendor": "nokia", "domain": "ran", "format": "json_restconf",
        "description": "Nokia gNB — 5G NR cell out of service (root cause: transport failure upstream)"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# NOKIA 1830 PSS — OPTICAL
# ─────────────────────────────────────────────────────────────────────────────
NOKIA_OPTICAL_SAMPLES: List[Dict[str, Any]] = [
    {
        "payload": {
            "alarmSequenceNo": 9876543,
            "raisedTime": "2024-01-15T14:22:00Z",
            "alarmSeverity": "CRITICAL",
            "alarmCondition": "LOS",
            "serviceAffecting": True,
            "neName": "PSS-32-NODE-A",
            "neType": "1830PSS32",
            "facilityType": "OCH",
            "facilityName": "OCH-1-1-1-TX",
            "conditionDescription": "Loss of Signal on Optical Channel OCH-1-1-1-TX",
            "additionalInfo": "Rx Power: -40.0 dBm (threshold: -28.0 dBm); Span: Mumbai-DC1 to Chennai-DC2",
            "clearTime": None
        },
        "vendor": "nokia", "domain": "optical", "format": "json_restconf",
        "description": "Nokia 1830PSS — optical Loss of Signal on OCH (fiber cut or amplifier failure)"
    },
    {
        "payload": {
            "alarmSequenceNo": 9876544,
            "raisedTime": "2024-01-15T14:22:05Z",
            "alarmSeverity": "CRITICAL",
            "alarmCondition": "AMPLIFIER_FAULT",
            "serviceAffecting": True,
            "neName": "PSS-32-NODE-A",
            "neType": "1830PSS32",
            "facilityType": "AMP",
            "facilityName": "AMP-1-3-1",
            "conditionDescription": "EDFA amplifier AMP-1-3-1 output power below threshold",
            "additionalInfo": "Output power: -35 dBm; Expected: +3 dBm; Pump laser failure suspected",
            "clearTime": None
        },
        "vendor": "nokia", "domain": "optical", "format": "json_restconf",
        "description": "Nokia 1830PSS — EDFA amplifier hardware fault (pump laser failure)"
    },
    {
        "payload": {
            "alarmSequenceNo": 9876545,
            "raisedTime": "2024-01-15T14:21:50Z",
            "alarmSeverity": "MAJOR",
            "alarmCondition": "OSNR_DEGRADATION",
            "serviceAffecting": True,
            "neName": "PSS-32-NODE-B",
            "neType": "1830PSS32",
            "facilityType": "OCH",
            "facilityName": "OCH-2-1-1-RX",
            "conditionDescription": "OSNR below threshold on received optical channel",
            "additionalInfo": "OSNR: 12.3 dB (threshold: 18.0 dB); likely span loss increase",
            "clearTime": None
        },
        "vendor": "nokia", "domain": "optical", "format": "json_restconf",
        "description": "Nokia 1830PSS — OSNR degradation (pre-cursor to LOS; fiber bend or connector issue)"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# ERICSSON ENM — RAN
# ─────────────────────────────────────────────────────────────────────────────
ERICSSON_ENM_SAMPLES: List[Dict[str, Any]] = [
    {
        "payload": {
            "alarmId": 20240115001234,
            "objectOfFailure": "SubNetwork=Europe,MeContext=RBS-SITE-001,ManagedElement=1,ENodeBFunction=1",
            "specificProblem": "Cell Disabled",
            "probableCause": "CELL_DISABLED",
            "perceivedSeverity": "A1",
            "alarmText": "E-UTRAN cell LTE-CELL-001A has been administratively disabled. 32 subscribers affected.",
            "eventTime": "2024-01-15T14:23:01.456Z",
            "ceaseTime": None,
            "alarmingObject": "EUtranCellFDD=LTE-CELL-001A",
            "alarmingObjectType": "EUtranCellFDD",
            "mangedObjectType": "ENodeBFunction",
            "insertTime": "2024-01-15T14:23:05.000Z",
            "ackStatus": "UNACKNOWLEDGED",
            "serviceAffecting": "SA"
        },
        "vendor": "ericsson", "domain": "ran", "format": "json_rest",
        "description": "Ericsson eNB — LTE cell administratively disabled (A1/Critical)"
    },
    {
        "payload": {
            "alarmId": 20240115001235,
            "objectOfFailure": "SubNetwork=Europe,MeContext=RBS-SITE-001,ManagedElement=1,ENodeBFunction=1",
            "specificProblem": "Baseband Unit Hardware Fault",
            "probableCause": "BASEBAND_UNIT_FAULT",
            "perceivedSeverity": "A1",
            "alarmText": "Hardware fault detected in baseband unit BBU-01. Board temperature: 92°C. Affected cells: 3.",
            "eventTime": "2024-01-15T13:45:00.000Z",
            "ceaseTime": None,
            "alarmingObject": "BasebandUnit=BBU-01",
            "alarmingObjectType": "BasebandUnit",
            "mangedObjectType": "ENodeBFunction",
            "insertTime": "2024-01-15T13:45:04.000Z",
            "ackStatus": "ACKNOWLEDGED",
            "serviceAffecting": "SA"
        },
        "vendor": "ericsson", "domain": "ran", "format": "json_rest",
        "description": "Ericsson eNB — baseband unit hardware fault (A1/Critical; board overheating)"
    },
    {
        "payload": {
            "alarmId": 20240115001240,
            "objectOfFailure": "SubNetwork=Europe,MeContext=RBS-SITE-001,ManagedElement=1",
            "specificProblem": "Clock Synchronization Loss",
            "probableCause": "CLOCK_SYNC_LOSS",
            "perceivedSeverity": "A2",
            "alarmText": "GPS clock synchronization lost. Site falling back to free-running mode. Timing accuracy degraded.",
            "eventTime": "2024-01-15T12:30:00.000Z",
            "ceaseTime": None,
            "alarmingObject": "SynchronizationModule=SYNC-1",
            "alarmingObjectType": "SynchronizationModule",
            "mangedObjectType": "ManagedElement",
            "insertTime": "2024-01-15T12:30:03.000Z",
            "ackStatus": "UNACKNOWLEDGED",
            "serviceAffecting": "NSA"
        },
        "vendor": "ericsson", "domain": "ran", "format": "json_rest",
        "description": "Ericsson eNB — GPS/PTP clock sync loss (A2/Major; timing degradation)"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HUAWEI iMANAGER — IP Domain
# ─────────────────────────────────────────────────────────────────────────────
HUAWEI_IP_SAMPLES: List[Dict[str, Any]] = [
    {
        "payload": {
            "alarmId": "HW-2024011500056789",
            "deviceName": "HW-AGG-RTR-01",
            "deviceIp": "10.1.1.101",
            "alarmName": "Interface Down",
            "alarmLevel": "Critical",
            "alarmCategory": "Communication Alarm",
            "alarmSource": "GigabitEthernet0/0/1",
            "alarmReason": "The interface GigabitEthernet0/0/1 status changes to Down. Remote peer unreachable.",
            "alarmTime": "2024-01-15 14:23:01",
            "clearTime": None,
            "alarmCode": "ALM-3276",
            "locationInfo": "DC1-RACK-A3-Slot2",
            "additionalInfo": "ifIndex=5; ifDescr=GigabitEthernet0/0/1; ifSpeed=10G",
            "nmsId": "Huawei-U2000-PROD"
        },
        "vendor": "huawei", "domain": "ip", "format": "json_rest",
        "description": "Huawei aggregation router — 10G interface down (DC1 to optical POP backhaul)"
    },
    {
        "payload": {
            "alarmId": "HW-2024011500056790",
            "deviceName": "HW-AGG-RTR-01",
            "deviceIp": "10.1.1.101",
            "alarmName": "BGP Peer Down",
            "alarmLevel": "Major",
            "alarmCategory": "Communication Alarm",
            "alarmSource": "BGP process",
            "alarmReason": "BGP session to peer 10.2.2.1 has gone down. Reason: hold-timer expired.",
            "alarmTime": "2024-01-15 14:23:08",
            "clearTime": None,
            "alarmCode": "ALM-4501",
            "locationInfo": "DC1-RACK-A3",
            "additionalInfo": "PeerIP=10.2.2.1; PeerAS=65002; LocalAS=65001",
            "nmsId": "Huawei-U2000-PROD"
        },
        "vendor": "huawei", "domain": "ip", "format": "json_rest",
        "description": "Huawei router — BGP session down (cascades from interface failure)"
    },
    {
        "payload": {
            "alarmId": "HW-2024011500056800",
            "deviceName": "HW-CORE-RTR-02",
            "deviceIp": "10.0.0.2",
            "alarmName": "CPU Utilization High",
            "alarmLevel": "Major",
            "alarmCategory": "Quality of Service Alarm",
            "alarmSource": "CPU",
            "alarmReason": "CPU utilization has exceeded 90% for over 5 minutes.",
            "alarmTime": "2024-01-15 15:45:00",
            "clearTime": None,
            "alarmCode": "ALM-1102",
            "locationInfo": "DC1-RACK-C1",
            "additionalInfo": "CPU5sec=92%; CPU1min=88%; CPU5min=85%; TopProcess=BGP",
            "nmsId": "Huawei-U2000-PROD"
        },
        "vendor": "huawei", "domain": "ip", "format": "json_rest",
        "description": "Huawei core router — CPU utilisation >90% (BGP churn suspected)"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# SNMP TRAPS — Generic (any vendor sending IF-MIB traps)
# ─────────────────────────────────────────────────────────────────────────────
SNMP_TRAP_SAMPLES: List[Dict[str, Any]] = [
    {
        "payload": {
            "source_ip": "192.168.1.50",
            "community": "public",
            "enterprise_oid": "1.3.6.1.2.1.2.2.1",
            "trap_type": "linkDown",
            "agent_address": "192.168.1.50",
            "timestamp": "2024-01-15T14:23:02Z",
            "varbinds": {
                "sysName": "dc1-access-sw-07",
                "ifIndex": "12",
                "ifDescr": "TenGigabitEthernet1/0/12",
                "ifAdminStatus": "1",
                "ifOperStatus": "2",
                "ifAlias": "TO-SERVER-RACK-B"
            }
        },
        "vendor": "generic", "domain": "ip", "format": "snmp_trap",
        "description": "Generic IF-MIB linkDown trap from DC access switch (server uplink)"
    },
    {
        "payload": {
            "source_ip": "10.2.3.4",
            "community": "monitoring",
            "enterprise_oid": "1.3.6.1.6.3.1.1.5.4",
            "trap_type": "linkUp",
            "agent_address": "10.2.3.4",
            "timestamp": "2024-01-15T14:25:30Z",
            "varbinds": {
                "sysName": "dc1-access-sw-07",
                "ifIndex": "12",
                "ifDescr": "TenGigabitEthernet1/0/12",
                "ifAdminStatus": "1",
                "ifOperStatus": "1",
                "ifAlias": "TO-SERVER-RACK-B"
            }
        },
        "vendor": "generic", "domain": "ip", "format": "snmp_trap",
        "description": "IF-MIB linkUp — interface restored (clearing the linkDown above)"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS ALERTMANAGER — Compute / Cloud
# ─────────────────────────────────────────────────────────────────────────────
PROMETHEUS_SAMPLES: List[Dict[str, Any]] = [
    {
        "payload": {
            "alerts": [{
                "status": "firing",
                "labels": {
                    "alertname": "HostDown",
                    "severity": "critical",
                    "instance": "compute-node-07:9100",
                    "job": "node_exporter",
                    "datacenter": "DC1",
                    "rack": "C3"
                },
                "annotations": {
                    "summary": "Host compute-node-07 is unreachable",
                    "description": "compute-node-07 has been down for more than 5 minutes. All VMs on this host are affected."
                },
                "startsAt": "2024-01-15T14:20:00.000Z",
                "endsAt": "0001-01-01T00:00:00Z",
                "fingerprint": "a1b2c3d4e5f6"
            }]
        },
        "vendor": "prometheus", "domain": "compute", "format": "webhook_json",
        "description": "Prometheus — bare-metal host down (hypervisor failure; VMs impacted)"
    },
    {
        "payload": {
            "alerts": [{
                "status": "firing",
                "labels": {
                    "alertname": "DiskFull",
                    "severity": "major",
                    "instance": "compute-node-03:9100",
                    "job": "node_exporter",
                    "mountpoint": "/data",
                    "device": "/dev/sdb"
                },
                "annotations": {
                    "summary": "Disk /data on compute-node-03 is 95% full",
                    "description": "Disk utilization on /dev/sdb has reached 95%. Write operations will fail when 100% is reached."
                },
                "startsAt": "2024-01-15T13:10:00.000Z",
                "endsAt": "0001-01-01T00:00:00Z",
                "fingerprint": "b2c3d4e5f6a7"
            }]
        },
        "vendor": "prometheus", "domain": "compute", "format": "webhook_json",
        "description": "Prometheus — disk nearly full on compute node (storage capacity alarm)"
    },
    {
        "payload": {
            "alerts": [{
                "status": "resolved",
                "labels": {
                    "alertname": "HostDown",
                    "severity": "critical",
                    "instance": "compute-node-07:9100",
                    "job": "node_exporter",
                },
                "annotations": {
                    "summary": "Host compute-node-07 is back online",
                    "description": "compute-node-07 recovered after power cycle."
                },
                "startsAt": "2024-01-15T14:20:00.000Z",
                "endsAt": "2024-01-15T14:45:00.000Z",
                "fingerprint": "a1b2c3d4e5f6"
            }]
        },
        "vendor": "prometheus", "domain": "compute", "format": "webhook_json",
        "description": "Prometheus — host restored (clearing event for HostDown)"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# KUBERNETES EVENTS
# ─────────────────────────────────────────────────────────────────────────────
KUBERNETES_SAMPLES: List[Dict[str, Any]] = [
    {
        "payload": {
            "apiVersion": "v1",
            "kind": "Event",
            "metadata": {"name": "upf-pod-abc.1729abc", "namespace": "ran-functions"},
            "involvedObject": {
                "kind": "Pod",
                "name": "upf-pod-0",
                "namespace": "ran-functions"
            },
            "reason": "BackOff",
            "message": "Back-off restarting failed container upf-main in pod upf-pod-0. CrashLoopBackOff.",
            "type": "Warning",
            "firstTimestamp": "2024-01-15T14:18:00Z",
            "lastTimestamp": "2024-01-15T14:23:00Z",
            "count": 12,
            "source": {"component": "kubelet", "host": "worker-node-3"}
        },
        "vendor": "kubernetes", "domain": "compute", "format": "k8s_event",
        "description": "K8s — UPF pod in CrashLoopBackOff (5G core user plane function unavailable)"
    },
    {
        "payload": {
            "apiVersion": "v1",
            "kind": "Event",
            "metadata": {"name": "amf-pod.173babc", "namespace": "5g-core"},
            "involvedObject": {
                "kind": "Pod",
                "name": "amf-pod-1",
                "namespace": "5g-core"
            },
            "reason": "OOMKilling",
            "message": "OOM killer invoked on container amf-main. Memory limit: 4Gi, usage at time of kill: 4.1Gi",
            "type": "Warning",
            "firstTimestamp": "2024-01-15T13:55:00Z",
            "lastTimestamp": "2024-01-15T13:55:01Z",
            "count": 1,
            "source": {"component": "kubelet", "host": "worker-node-1"}
        },
        "vendor": "kubernetes", "domain": "compute", "format": "k8s_event",
        "description": "K8s — AMF pod OOM killed (5G core control plane function impacted)"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Combined batch — representative cross-domain incident scenario
# (optical fiber cut cascading through IP to RAN)
# ─────────────────────────────────────────────────────────────────────────────
FIBER_CUT_CASCADE_SCENARIO: List[Dict[str, Any]] = [
    # T+0: Optical OSNR degradation (early warning)
    NOKIA_OPTICAL_SAMPLES[2],
    # T+10s: Optical LOS (fiber cut confirmed)
    NOKIA_OPTICAL_SAMPLES[0],
    # T+10s: Optical amplifier fault (same span)
    NOKIA_OPTICAL_SAMPLES[1],
    # T+15s: Cisco router syslog — interface down (IP circuit on this fiber)
    CISCO_SYSLOG_SAMPLES[0],
    # T+20s: BGP peer down (cascades from interface)
    CISCO_SYSLOG_SAMPLES[1],
    # T+22s: Nokia gNB — cell out of service (backhaul failure)
    NOKIA_RAN_SAMPLES[2],
    # T+25s: Nokia gNB — CPRI failure (transport lost)
    NOKIA_RAN_SAMPLES[1],
]

ALL_SAMPLES: List[Dict[str, Any]] = (
    CISCO_SYSLOG_SAMPLES
    + NOKIA_RAN_SAMPLES
    + NOKIA_OPTICAL_SAMPLES
    + ERICSSON_ENM_SAMPLES
    + HUAWEI_IP_SAMPLES
    + SNMP_TRAP_SAMPLES
    + PROMETHEUS_SAMPLES
    + KUBERNETES_SAMPLES
)
