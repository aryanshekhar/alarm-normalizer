import { useRef, useState, useLayoutEffect, useCallback, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

// ── India bounding box ────────────────────────────────────────────────────────
const LAT_MIN = 8,  LAT_MAX = 37;
const LON_MIN = 68, LON_MAX = 97;
const MAP_H   = 340;
const PAD     = 28;

// Fallback centre of India for nodes with no coordinates
const FALLBACK_LAT = 22;
const FALLBACK_LON = 82.5;

function project(lat, lon, width) {
  const x = PAD + ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * (width - 2 * PAD);
  const y = PAD + ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * (MAP_H - 2 * PAD);
  return { x, y };
}

// ── Deterministic jitter so co-located fallback nodes don't fully overlap ─────
function jitter(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (Math.imul(31, h) + id.charCodeAt(i)) | 0;
  return { dx: ((h & 0xff) - 128) * 0.15, dy: (((h >> 8) & 0xff) - 128) * 0.15 };
}

// ── Node classification ───────────────────────────────────────────────────────
// Checks node.type (a stored property) first, then node.labels[0] (Neo4j label)
function nodeStyle(node) {
  const t = (node.type ?? '').toUpperCase();
  const l = node.labels?.[0] ?? '';

  if (t === 'P_ROUTER')                       return { color: '#ef4444', group: 'Core Router' };
  if (t === 'PE_ROUTER')                      return { color: '#f97316', group: 'PE Router'   };
  if (t === 'AGG_SWITCH')                     return { color: '#fb923c', group: 'Agg Switch'  };
  if (t === 'ROADM' || t === 'OTN_TRANSPONDER' || t === 'AMPLIFIER')
                                              return { color: '#3b82f6', group: 'Optical'     };
  if (l === 'OpticalNode')                    return { color: '#3b82f6', group: 'Optical'     };
  if (t === 'GNB' || l === 'RANNode')         return { color: '#22c55e', group: 'gNB / RAN'   };
  if (l === 'Cell')                           return { color: '#86efac', group: 'Cell'        };
  if (t === 'PHYSICAL_HOST' || l === 'Host')  return { color: '#a855f7', group: 'Host'        };
  if (['AMF','SMF','UPF','PCF','AUSF','NRF'].includes(t) || l === 'VNF')
                                              return { color: '#c084fc', group: 'VNF'         };
  if (l === 'Service')                        return { color: '#f59e0b', group: 'Service'     };
  if (l === 'NetworkSlice')                   return { color: '#fbbf24', group: 'Net Slice'   };
  if (l === 'Alarm')                          return { color: '#ef4444', group: 'Alarm'       };
  // Catch-all — visible gray so nothing disappears
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

const GRID_LATS = [10, 15, 20, 25, 30, 35];
const GRID_LONS = [70, 75, 80, 85, 90, 95];

// ── Component ─────────────────────────────────────────────────────────────────
export default function TopologyMap({ topology }) {
  const containerRef  = useRef(null);
  const [width, setWidth] = useState(600);
  const [locked, setLocked] = useState(true);
  // Remembers user-dragged positions so re-locking snaps to them
  const draggedPos = useRef({});

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (!topology?.nodes) return { nodes: [], links: [] };

    // Debug: log unique label/type combos so the console shows real API data
    const seen = new Set();
    topology.nodes.forEach((n) => {
      const key = `label=${n.labels?.[0] ?? '?'}  type=${n.type ?? '?'}`;
      if (!seen.has(key)) { seen.add(key); }
    });
    console.log('[TopologyMap] unique label/type pairs:', [...seen].sort().join('\n'));

    const nodes = topology.nodes.map((n) => {
      const hasGeo = n.latitude != null && n.longitude != null;
      const lat    = hasGeo ? n.latitude  : FALLBACK_LAT;
      const lon    = hasGeo ? n.longitude : FALLBACK_LON;
      const { x: baseX, y: baseY } = project(lat, lon, width);
      const { dx, dy } = hasGeo ? { dx: 0, dy: 0 } : jitter(n.id ?? '');
      const x = baseX + dx;
      const y = baseY + dy;

      if (locked) {
        // Use user-dragged position if available, else projected
        const saved = draggedPos.current[n.id];
        return { ...n, fx: saved?.x ?? x, fy: saved?.y ?? y };
      }
      // Unlocked: seed x/y hint but no pin — node is draggable
      return { ...n, x, y, fx: undefined, fy: undefined };
    });

    const ids   = new Set(nodes.map((n) => n.id));
    const links = (topology.edges ?? [])
      .filter((e) => ids.has(e.from) && ids.has(e.to))
      .map((e) => ({ source: e.from, target: e.to }));

    return { nodes, links };
  }, [topology, width, locked]);

  const drawGrid = useCallback((ctx) => {
    ctx.save();
    ctx.strokeStyle = 'rgba(55, 65, 81, 0.55)';
    ctx.lineWidth   = 0.4;
    ctx.setLineDash([3, 5]);
    ctx.fillStyle   = 'rgba(107, 114, 128, 0.75)';
    ctx.font        = '3.5px ui-monospace, monospace';

    for (const lat of GRID_LATS) {
      const { x: x0, y } = project(lat, LON_MIN, width);
      const { x: x1 }    = project(lat, LON_MAX, width);
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillText(`${lat}°N`, x0 - 10, y + 1.2);
    }
    for (const lon of GRID_LONS) {
      const { x, y: y0 } = project(LAT_MAX, lon, width);
      const { y: y1 }    = project(LAT_MIN, lon, width);
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
      ctx.fillText(`${lon}°E`, x - 2, y1 + 6);
    }
    ctx.restore();
  }, [width]);

  const paintNode = useCallback((node, ctx, globalScale) => {
    const { color } = nodeStyle(node);
    const r = Math.max(2, 5 / globalScale);
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.45)';
    ctx.lineWidth   = 0.5 / globalScale;
    ctx.stroke();
  }, []);

  const onNodeDragEnd = useCallback((node) => {
    draggedPos.current[node.id] = { x: node.x, y: node.y };
  }, []);

  if (!topology?.nodes?.length) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 h-72 flex items-center justify-center">
        <p className="text-gray-600 text-sm">Load topology to see network graph</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2 border-b border-gray-800 flex items-center justify-between">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Network Topology
        </h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">
            {graphData.nodes.length} nodes &middot; {graphData.links.length} links
          </span>
          {/* Lock / unlock toggle */}
          <button
            onClick={() => setLocked((l) => !l)}
            title={locked ? 'Unlock nodes for manual drag' : 'Lock nodes to geo positions'}
            className={`text-xs px-2 py-0.5 rounded border transition-colors ${
              locked
                ? 'border-blue-700 bg-blue-950 text-blue-300 hover:bg-blue-900'
                : 'border-yellow-700 bg-yellow-950 text-yellow-300 hover:bg-yellow-900'
            }`}
          >
            {locked ? '🔒 Locked' : '🔓 Unlock'}
          </button>
        </div>
      </div>

      {/* Map + legend row */}
      <div className="flex">
        {/* Canvas */}
        <div
          ref={containerRef}
          className="flex-1 min-w-0 overflow-hidden"
          style={{ height: MAP_H }}
        >
          <ForceGraph2D
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
            cooldownTicks={locked ? 0 : 80}
            d3AlphaDecay={locked ? 1 : 0.03}
            d3VelocityDecay={locked ? 1 : 0.4}
            enableNodeDrag={!locked}
            onNodeDragEnd={onNodeDragEnd}
            enableZoomInteraction
            onRenderFramePre={drawGrid}
          />
        </div>

        {/* Legend — explicit z-index so it sits above the canvas */}
        <div
          className="w-28 shrink-0 px-3 py-3 border-l border-gray-800 space-y-1.5 bg-gray-900"
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
