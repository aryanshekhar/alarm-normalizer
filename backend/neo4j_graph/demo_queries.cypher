// ============================================================
// DEMO CYPHER QUERIES — Telecom Topology + Alarm Graph
// Run these in the Neo4j Browser at http://YOUR_GCP_IP:7474
// ============================================================


// ────────────────────────────────────────────────────────────
// 1.  FULL GRAPH OVERVIEW
//     Best opening query for a client demo.
//     Shows everything in one canvas.
// ────────────────────────────────────────────────────────────
MATCH (n) RETURN n;


// ────────────────────────────────────────────────────────────
// 2.  TOPOLOGY ONLY (no alarms, no cells)
//     Clean view of the network infrastructure layers.
// ────────────────────────────────────────────────────────────
MATCH (n:NetworkNode)-[r]->(m:NetworkNode)
RETURN n, r, m;


// ────────────────────────────────────────────────────────────
// 3.  OPTICAL LAYER ONLY
//     ROADMs, amplifiers, OTN transponders and fiber spans.
// ────────────────────────────────────────────────────────────
MATCH (n:OpticalNode)-[r]->(m)
RETURN n, r, m;


// ────────────────────────────────────────────────────────────
// 4.  FIBER CUT CASCADE — root cause to symptoms
//     The most compelling demo query.
//     Shows how one optical LOS cascades through IP into RAN.
// ────────────────────────────────────────────────────────────
MATCH path = (root:Alarm {isRootCause: true, id: 'ALM-OPT-001'})
              -[:PROPAGATED_TO*1..4]->(symptom:Alarm)
RETURN path;


// ────────────────────────────────────────────────────────────
// 5.  ALL ACTIVE ALARMS with their affected devices
//     Shows every raised alarm and the node it fired on.
// ────────────────────────────────────────────────────────────
MATCH (a:Alarm)-[:TRIGGERED_ON]->(n:NetworkNode)
WHERE a.state = 'raised'
RETURN a.id            AS AlarmID,
       a.perceivedSeverity AS Severity,
       a.domain        AS Domain,
       a.vendor        AS Vendor,
       a.specificProblem   AS Problem,
       n.name          AS AffectedDevice,
       a.serviceAffecting  AS ServiceAffecting,
       a.isRootCause   AS IsRootCause
ORDER BY
  CASE a.perceivedSeverity
    WHEN 'critical' THEN 1
    WHEN 'major'    THEN 2
    WHEN 'minor'    THEN 3
    WHEN 'warning'  THEN 4
    ELSE 5
  END;


// ────────────────────────────────────────────────────────────
// 6.  SERVICE IMPACT — which services are affected right now
//     Shows business impact of current active alarms.
// ────────────────────────────────────────────────────────────
MATCH (a:Alarm)-[:AFFECTS_SERVICE]->(s:Service)
WHERE a.state = 'raised'
RETURN s.name          AS Service,
       s.priority      AS Priority,
       s.customer      AS Customer,
       collect(a.id)   AS AlarmIDs,
       collect(a.perceivedSeverity) AS Severities,
       count(a)        AS AlarmCount
ORDER BY s.priority;


// ────────────────────────────────────────────────────────────
// 7.  CROSS-DOMAIN ALARM VIEW — graph of alarms + topology
//     Shows alarms linked to devices across all four domains.
// ────────────────────────────────────────────────────────────
MATCH (a:Alarm)-[:TRIGGERED_ON]->(n:NetworkNode)
WHERE a.state = 'raised'
RETURN a, n;


// ────────────────────────────────────────────────────────────
// 8.  COMPUTE CASCADE — host failure into VNF
// ────────────────────────────────────────────────────────────
MATCH path = (root:Alarm {id: 'ALM-CMP-001'})
              -[:PROPAGATED_TO*]->(symptom:Alarm)
WITH path
MATCH (n1:Alarm)-[:TRIGGERED_ON]->(d1:NetworkNode)
WHERE n1 IN nodes(path)
RETURN path, d1;


// ────────────────────────────────────────────────────────────
// 9.  RAN BACKHAUL PATH — from gNB to optical
//     Shows the full dependency chain for a RAN site.
// ────────────────────────────────────────────────────────────
MATCH path = (g:NetworkNode {id: 'gNB-MUM-SITE-A01'})
              -[:BACKHAUL_VIA]->(pe:NetworkNode)
              -[:OPTICAL_CIRCUIT|IP_LINK*1..3]->(optical:NetworkNode)
RETURN path;


// ────────────────────────────────────────────────────────────
// 10. ALARM COUNT BY DOMAIN AND SEVERITY
//     Summary table — useful for dashboard view.
// ────────────────────────────────────────────────────────────
MATCH (a:Alarm)
WHERE a.state = 'raised'
RETURN a.domain        AS Domain,
       a.perceivedSeverity AS Severity,
       count(a)        AS Count
ORDER BY Domain, Severity;


// ────────────────────────────────────────────────────────────
// 11. DEVICES WITH MOST ALARMS
//     Identifies hot spots in the network.
// ────────────────────────────────────────────────────────────
MATCH (a:Alarm)-[:TRIGGERED_ON]->(n:NetworkNode)
RETURN n.name       AS Device,
       n.domain     AS Domain,
       count(a)     AS AlarmCount,
       collect(a.perceivedSeverity) AS Severities
ORDER BY AlarmCount DESC
LIMIT 10;


// ────────────────────────────────────────────────────────────
// 12. FULL PATH — alarm to service impact via topology
//     Shows end-to-end: root cause → propagation → service.
// ────────────────────────────────────────────────────────────
MATCH (root:Alarm {isRootCause: true})
      -[:PROPAGATED_TO*0..4]->(a:Alarm)
      -[:TRIGGERED_ON]->(n:NetworkNode)
OPTIONAL MATCH (a)-[:AFFECTS_SERVICE]->(s:Service)
RETURN root.id         AS RootCause,
       a.id            AS Alarm,
       a.domain        AS Domain,
       n.name          AS Device,
       s.name          AS ImpactedService
ORDER BY root.id;


// ────────────────────────────────────────────────────────────
// 13. SUPPRESS CHILD ALARMS — show only root causes
//     This is the alarm noise reduction use case.
// ────────────────────────────────────────────────────────────
MATCH (a:Alarm)
WHERE a.state = 'raised'
  AND NOT ()-[:PROPAGATED_TO]->(a)
RETURN a.id            AS RootCauseAlarmID,
       a.perceivedSeverity AS Severity,
       a.domain        AS Domain,
       a.specificProblem   AS Problem,
       a.vendor        AS Vendor;


// ────────────────────────────────────────────────────────────
// 14. NOKIA OPTICAL NODES AND THEIR ALARMS
//     Domain-specific view for optical operations team.
// ────────────────────────────────────────────────────────────
MATCH (n:OpticalNode)
OPTIONAL MATCH (a:Alarm)-[:TRIGGERED_ON]->(n)
WHERE a.state = 'raised'
RETURN n.name          AS Node,
       n.type          AS Type,
       n.site          AS Site,
       a.perceivedSeverity AS AlarmSeverity,
       a.specificProblem   AS Problem
ORDER BY n.city;


// ────────────────────────────────────────────────────────────
// 15. SLICE AND SERVICE HEALTH
//     Shows each slice, its services, and alarm status.
// ────────────────────────────────────────────────────────────
MATCH (sl:NetworkSlice)<-[:BELONGS_TO_SLICE]-(s:Service)
OPTIONAL MATCH (a:Alarm)-[:AFFECTS_SERVICE]->(s)
WHERE a.state = 'raised'
RETURN sl.name         AS Slice,
       sl.type         AS SliceType,
       s.name          AS Service,
       s.customer      AS Customer,
       count(a)        AS ActiveAlarms
ORDER BY sl.type, s.name;
