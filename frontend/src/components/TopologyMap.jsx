import { useRef, useState, useLayoutEffect, useCallback, useMemo, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

// ── Constants ─────────────────────────────────────────────────────────────────
const LS_KEY   = 'aiops-topology-layout';
const LAT_MIN  = 8,  LAT_MAX = 37;
const LON_MIN  = 68, LON_MAX = 97;
const MAP_H    = 340;
const PAD      = 28;
const FALLBACK_LAT = 22;
const FALLBACK_LON = 82.5;

const GRID_LATS = [10, 15, 20, 25, 30, 35];
const GRID_LONS = [70, 75, 80, 85, 90, 95];

// ── Projection ────────────────────────────────────────────────────────────────
// Maps lat/lon → graph-space (x, y). Width is the canvas CSS pixel width.
// Node fx/fy and grid-line coordinates share this same space, so they always
// align regardless of the current zoom/pan transform.
function project(lat, lon, width) {
  const x = PAD + ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * (width - 2 * PAD);
  const y = PAD + ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * (MAP_H - 2 * PAD);
  return { x, y };
}

// Deterministic per-ID jitter for nodes that share the fallback position
function jitter(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (Math.imul(31, h) + id.charCodeAt(i)) | 0;
  return { dx: ((h & 0xff) - 128) * 0.15, dy: (((h >> 8) & 0xff) - 128) * 0.15 };
}

// ── Position store helpers ────────────────────────────────────────────────────
function serializePos(posRef) {
  return JSON.stringify(
    Object.entries(posRef.current).map(([id, { x, y }]) => ({ id, x, y })),
  );
}

function parsePos(json) {
  const arr = JSON.parse(json);
  const map = {};
  for (const { id, x, y } of arr) {
    if (id && x != null && y != null) map[id] = { x, y };
  }
  return map;
}

// ── Node classification ───────────────────────────────────────────────────────
function nodeStyle(node) {
  const t = (node.type ?? '').toUpperCase();
  const l = node.labels?.[0] ?? '';

  if (t === 'P_ROUTER')                                    return { color: '#ef4444', group: 'Core Router' };
  if (t === 'PE_ROUTER')                                   return { color: '#f97316', group: 'PE Router'   };
  if (t === 'AGG_SWITCH')                                  return { color: '#fb923c', group: 'Agg Switch'  };
  if (['ROADM', 'OTN_TRANSPONDER', 'AMPLIFIER'].includes(t) || l === 'OpticalNode')
                                                           return { color: '#3b82f6', group: 'Optical'     };
  if (t === 'GNB' || l === 'RANNode')                      return { color: '#22c55e', group: 'gNB / RAN'   };
  if (l === 'Cell')                                        return { color: '#86efac', group: 'Cell'        };
  if (t === 'PHYSICAL_HOST' || l === 'Host')               return { color: '#a855f7', group: 'Host'        };
  if (['AMF','SMF','UPF','PCF','AUSF','NRF'].includes(t) || l === 'VNF')
                                                           return { color: '#c084fc', group: 'VNF'         };
  if (l === 'Service')                                     return { color: '#f59e0b', group: 'Service'     };
  if (l === 'NetworkSlice')                                return { color: '#fbbf24', group: 'Net Slice'   };
  if (l === 'Alarm')                                       return { color: '#f87171', group: 'Alarm'       };
  return { color: '#9ca3af', group: l || t || 'Other' };
}

const LEGEND = [
  { color: '#ef4444', name: 'Core Router' },
  { color: '#f97316', name: 'PE Router'   },
  { color: '#fb923c', name: 'Agg Switch'  },
  { color: '#3b82f6', name: 'Optical'     },
  { color: '#22c55e', name: 'gNB / RAN'   },
  { color: '#86efac', name: 'Cell'        },
  { color: '#a855f7', name: 'Host'        },
  { color: '#c084fc', name: 'VNF'         },
  { color: '#f59e0b', name: 'Service'     },
  { color: '#fbbf24', name: 'Net Slice'   },
  { color: '#9ca3af', name: 'Other'       },
];

// ── Component ─────────────────────────────────────────────────────────────────
export default function TopologyMap({
  topology,
  alarmingNodeIds   = new Set(),
  rootCauseNodeIds  = new Set(),
  suppressedNodeIds = new Set(),
}) {
  const containerRef  = useRef(null);
  const fileInputRef  = useRef(null);
  const graphRef      = useRef(null);
  const [width, setWidth]       = useState(600);
  const [locked, setLocked]     = useState(true);
  const [pulsePhase, setPulsePhase] = useState(0);
  // posRef: { [nodeId]: { x, y } } — single source of truth for positions.
  // Populated from localStorage on mount, updated on drag / load.
  const posRef = useRef({});
  // Incrementing this forces graphData useMemo to recompute after a load.
  const [posVersion, setPosVersion] = useState(0);

  // Load saved positions from localStorage on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (raw) {
        posRef.current = parsePos(raw);
        setPosVersion((v) => v + 1);
      }
    } catch {/* ignore parse/storage errors */}
  }, []);

  // Pulse animation for alarming / root-cause nodes — tick every 600 ms
  useEffect(() => {
    if (alarmingNodeIds.size === 0 && rootCauseNodeIds.size === 0) return;
    const id = setInterval(() => {
      setPulsePhase((p) => p + 1);
      graphRef.current?.refresh();
    }, 600);
    return () => clearInterval(id);
  }, [alarmingNodeIds, rootCauseNodeIds]);

  // Track canvas container width
  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // ── Save layout ───────────────────────────────────────────────────────────
  const saveLayout = useCallback(() => {
    const json = serializePos(posRef);
    try { localStorage.setItem(LS_KEY, json); } catch {/* quota */}
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    Object.assign(document.createElement('a'), {
      href: url, download: 'topology-layout.json',
    }).click();
    URL.revokeObjectURL(url);
  }, []);

  // ── Load layout ───────────────────────────────────────────────────────────
  const loadLayout = useCallback(() => fileInputRef.current?.click(), []);

  const onFileChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        posRef.current = { ...posRef.current, ...parsePos(ev.target.result) };
        try { localStorage.setItem(LS_KEY, serializePos(posRef)); } catch {/* quota */}
        setPosVersion((v) => v + 1);
      } catch (err) {
        console.error('[TopologyMap] failed to load layout:', err);
      }
    };
    reader.readAsText(file);
    e.target.value = ''; // allow re-selecting same file
  }, []);

  // ── Lock toggle — auto-save to localStorage when locking ─────────────────
  const toggleLock = useCallback(() => {
    setLocked((prev) => {
      if (!prev) {
        // Transitioning unlocked → locked: persist positions
        try { localStorage.setItem(LS_KEY, serializePos(posRef)); } catch {/* quota */}
      }
      return !prev;
    });
  }, []);

  // ── Graph data ────────────────────────────────────────────────────────────
  const graphData = useMemo(() => {
    if (!topology?.nodes) return { nodes: [], links: [] };

    // Debug: print every unique label / type pair so we can verify mapping
    const seen = new Set();
    topology.nodes.forEach((n) => seen.add(`label=${n.labels?.[0] ?? '?'}  type=${n.type ?? '?'}`));
    console.log('[TopologyMap] label/type pairs:\n', [...seen].sort().join('\n'));

    const nodes = topology.nodes.map((n) => {
      // 1. Prefer manually set / loaded position
      const saved = posRef.current[n.id];
      if (saved) return { ...n, fx: saved.x, fy: saved.y };

      // 2. Project from lat/lon; fallback to centre India + jitter
      const hasGeo  = n.latitude != null && n.longitude != null;
      const lat     = hasGeo ? n.latitude  : FALLBACK_LAT;
      const lon     = hasGeo ? n.longitude : FALLBACK_LON;
      const { x: bx, y: by }  = project(lat, lon, width);
      const { dx, dy }        = hasGeo ? { dx: 0, dy: 0 } : jitter(n.id ?? '');
      return { ...n, fx: bx + dx, fy: by + dy };
    });

    const ids   = new Set(nodes.map((n) => n.id));
    const links = (topology.edges ?? [])
      .filter((e) => ids.has(e.from) && ids.has(e.to))
      .map((e) => ({ source: e.from, target: e.to }));

    return { nodes, links };
    // posVersion triggers recompute when positions are loaded from file/LS
  }, [topology, width, posVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Drag end — pin node at dropped position ───────────────────────────────
  const onNodeDragEnd = useCallback((node) => {
    // force-graph already sets node.fx / node.fy to the drop coordinates.
    // Mirror them into posRef so next graphData recompute preserves them.
    node.fx = node.x;
    node.fy = node.y;
    posRef.current[node.id] = { x: node.x, y: node.y };
  }, []);

  // ── Grid lines ────────────────────────────────────────────────────────────
  // onRenderFramePre receives a canvas ctx that has already been transformed
  // by d3-zoom (same transform applied to nodes). Drawing at project(lat,lon)
  // coordinates therefore moves with the graph on pan/zoom — NOT a DOM overlay.
  const drawGrid = useCallback(
    (ctx) => {
      ctx.save();
      ctx.strokeStyle = 'rgba(55, 65, 81, 0.55)';
      ctx.lineWidth   = 0.4;
      ctx.setLineDash([3, 5]);
      ctx.fillStyle   = 'rgba(107, 114, 128, 0.75)';
      ctx.font        = '3.5px ui-monospace, monospace';
      ctx.textBaseline = 'middle';

      for (const lat of GRID_LATS) {
        const { x: x0, y } = project(lat, LON_MIN, width);
        const { x: x1 }    = project(lat, LON_MAX, width);
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        ctx.fillText(`${lat}°N`, x0 - 12, y);
      }
      for (const lon of GRID_LONS) {
        const { x, y: y0 } = project(LAT_MAX, lon, width);
        const { y: y1 }    = project(LAT_MIN, lon, width);
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        ctx.fillText(`${lon}°E`, x - 3, y1 + 7);
      }
      ctx.restore();
    },
    [width],
  );

  // ── Node painter ──────────────────────────────────────────────────────────
  const paintNode = useCallback((node, ctx, globalScale) => {
    const { color } = nodeStyle(node);
    const r            = Math.max(2, 5 / globalScale);
    const isRootCause  = rootCauseNodeIds.has(node.id);
    const isSuppressed = suppressedNodeIds.has(node.id);
    // Plain red pulse for generic alarming (before differentiation is known)
    const isAlarming   = alarmingNodeIds.has(node.id) && !isRootCause && !isSuppressed;

    if (isRootCause) {
      // Bright red pulsing halo (larger than plain alarming)
      const pulse  = 0.5 + 0.5 * Math.sin(pulsePhase * Math.PI * 0.8);
      const outerR = r * (2.5 + 1.0 * pulse);
      ctx.beginPath();
      ctx.arc(node.x, node.y, outerR, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(239, 68, 68, ${0.30 + 0.25 * pulse})`;
      ctx.fill();
    } else if (isAlarming) {
      const pulse  = 0.5 + 0.5 * Math.sin(pulsePhase * Math.PI * 0.8);
      const outerR = r * (2.0 + 0.8 * pulse);
      ctx.beginPath();
      ctx.arc(node.x, node.y, outerR, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(239, 68, 68, ${0.20 + 0.20 * pulse})`;
      ctx.fill();
    }

    const fillColor   = isRootCause  ? '#ef4444'
                      : isSuppressed ? '#f97316'
                      : isAlarming   ? '#ef4444'
                      : color;
    const strokeColor = isRootCause  ? 'rgba(239,68,68,1.0)'
                      : isSuppressed ? 'rgba(249,115,22,0.8)'
                      : isAlarming   ? 'rgba(239,68,68,0.8)'
                      : 'rgba(0,0,0,0.45)';
    const strokeW     = (isRootCause ? 1.5 : isAlarming ? 1.2 : isSuppressed ? 1.0 : 0.5) / globalScale;

    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle   = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth   = strokeW;
    ctx.stroke();
  }, [alarmingNodeIds, rootCauseNodeIds, suppressedNodeIds, pulsePhase]);

  if (!topology?.nodes?.length) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 h-72 flex items-center justify-center">
        <p className="text-gray-600 text-sm">Load topology to see network graph</p>
      </div>
    );
  }

  const btnBase = 'text-xs px-2 py-0.5 rounded border transition-colors';

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      {/* Header ──────────────────────────────────────────────────────────── */}
      <div className="px-4 py-2 border-b border-gray-800 flex items-center justify-between gap-2 flex-wrap">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Network Topology
        </h2>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-gray-500">
            {graphData.nodes.length}&thinsp;nodes &middot; {graphData.links.length}&thinsp;links
          </span>

          {/* Save layout */}
          <button
            onClick={saveLayout}
            title="Download node positions as JSON and save to localStorage"
            className={`${btnBase} border-gray-600 bg-gray-800 text-gray-300 hover:bg-gray-700`}
          >
            💾 Save
          </button>

          {/* Load layout */}
          <button
            onClick={loadLayout}
            title="Load node positions from a JSON file"
            className={`${btnBase} border-gray-600 bg-gray-800 text-gray-300 hover:bg-gray-700`}
          >
            📂 Load
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            className="hidden"
            onChange={onFileChange}
          />

          {/* Lock / unlock */}
          <button
            onClick={toggleLock}
            title={locked ? 'Unlock to drag nodes' : 'Lock — auto-saves positions'}
            className={`${btnBase} ${
              locked
                ? 'border-blue-700 bg-blue-950 text-blue-300 hover:bg-blue-900'
                : 'border-yellow-700 bg-yellow-950 text-yellow-300 hover:bg-yellow-900'
            }`}
          >
            {locked ? '🔒 Locked' : '🔓 Unlocked'}
          </button>
        </div>
      </div>

      {/* Map + legend ─────────────────────────────────────────────────────── */}
      <div className="flex">
        {/* Canvas */}
        <div
          ref={containerRef}
          className="flex-1 min-w-0 overflow-hidden"
          style={{ height: MAP_H }}
        >
          <ForceGraph2D
            ref={graphRef}
            graphData={graphData}
            width={width}
            height={MAP_H}
            backgroundColor="#111827"
            nodeCanvasObject={paintNode}
            nodeCanvasObjectMode={() => 'replace'}
            nodeLabel={(n) =>
              `${n.id}${n.name ? ` — ${n.name}` : ''} [${n.labels?.[0] ?? ''} / ${n.type ?? ''}]`
            }
            linkColor={() => 'rgba(75, 85, 99, 0.45)'}
            linkWidth={0.6}
            // Force simulation is ALWAYS off — nodes only move via drag or load
            cooldownTicks={0}
            d3AlphaDecay={1}
            d3VelocityDecay={1}
            enableNodeDrag={!locked}
            onNodeDragEnd={onNodeDragEnd}
            enableZoomInteraction
            onRenderFramePre={drawGrid}
          />
        </div>

        {/* Legend */}
        <div
          className="w-28 shrink-0 px-3 py-3 border-l border-gray-800 bg-gray-900 space-y-1.5"
          style={{ position: 'relative', zIndex: 10 }}
        >
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Legend</p>
          {LEGEND.map(({ color, name }) => (
            <div key={name} className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0 border border-black/30"
                style={{ background: color }}
              />
              <span className="text-xs text-gray-400">{name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
