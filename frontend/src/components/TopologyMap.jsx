import { useRef, useState, useLayoutEffect, useCallback, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

// ── India bounding box ────────────────────────────────────────────────────────
const LAT_MIN = 8, LAT_MAX = 37;
const LON_MIN = 68, LON_MAX = 97;
const MAP_H   = 340;
const PAD     = 28;  // canvas padding in graph units

// Simple cylindrical equidistant projection (good enough for India's extent)
function project(lat, lon, width) {
  const x = PAD + ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * (width - 2 * PAD);
  const y = PAD + ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * (MAP_H - 2 * PAD);
  return { x, y };
}

// ── Node classification ───────────────────────────────────────────────────────
function nodeStyle(node) {
  const t = node.type ?? '';
  const l = node.labels?.[0] ?? '';
  if (t === 'P_ROUTER')                     return { color: '#ef4444', group: 'Core Router' };
  if (t === 'PE_ROUTER')                    return { color: '#f97316', group: 'PE Router' };
  if (t === 'AGG_SWITCH')                   return { color: '#fb923c', group: 'Agg Switch' };
  if (l === 'OpticalNode')                  return { color: '#3b82f6', group: 'Optical' };
  if (t === 'gNB' || l === 'RANNode')       return { color: '#22c55e', group: 'gNB / RAN' };
  if (l === 'Cell')                         return { color: '#86efac', group: 'Cell' };
  if (l === 'Host' || l === 'VNF')          return { color: '#a855f7', group: 'Host / VNF' };
  if (l === 'Service' || l === 'NetworkSlice') return { color: '#f59e0b', group: 'Service' };
  return { color: '#6b7280', group: l || t || 'Other' };
}

const LEGEND = [
  { color: '#ef4444', name: 'Core Router' },
  { color: '#f97316', name: 'PE Router'   },
  { color: '#3b82f6', name: 'Optical'     },
  { color: '#22c55e', name: 'gNB / RAN'   },
  { color: '#86efac', name: 'Cell'        },
  { color: '#a855f7', name: 'Host / VNF'  },
  { color: '#f59e0b', name: 'Service'     },
];

// Grid lat/lon lines drawn over the canvas
const GRID_LATS = [10, 15, 20, 25, 30, 35];
const GRID_LONS = [70, 75, 80, 85, 90, 95];

// ── Component ─────────────────────────────────────────────────────────────────
export default function TopologyMap({ topology }) {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(600);

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(([entry]) => {
      setWidth(entry.contentRect.width);
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (!topology?.nodes) return { nodes: [], links: [] };
    const nodes = topology.nodes
      .filter((n) => n.latitude != null && n.longitude != null)
      .map((n) => {
        const { x, y } = project(n.latitude, n.longitude, width);
        return { ...n, fx: x, fy: y };
      });
    const ids = new Set(nodes.map((n) => n.id));
    const links = (topology.edges ?? [])
      .filter((e) => ids.has(e.from) && ids.has(e.to))
      .map((e) => ({ source: e.from, target: e.to }));
    return { nodes, links };
  }, [topology, width]);

  // Draw lat/lon grid before nodes — ctx is already in graph-space coordinates
  const drawGrid = useCallback(
    (ctx) => {
      ctx.save();
      ctx.strokeStyle = 'rgba(55, 65, 81, 0.6)';
      ctx.lineWidth   = 0.4;
      ctx.setLineDash([3, 5]);
      ctx.fillStyle   = 'rgba(107, 114, 128, 0.8)';
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
    },
    [width],
  );

  // Custom node painter — circle sized independently of zoom
  const paintNode = useCallback((node, ctx, globalScale) => {
    const { color } = nodeStyle(node);
    const r = Math.max(2, 5 / globalScale);
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.lineWidth   = 0.6 / globalScale;
    ctx.stroke();
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
          Network Topology — India
        </h2>
        <span className="text-xs text-gray-500">
          {graphData.nodes.length} nodes &middot; {graphData.links.length} links
        </span>
      </div>

      <div className="flex">
        {/* Canvas map */}
        <div ref={containerRef} className="flex-1 min-w-0" style={{ height: MAP_H }}>
          <ForceGraph2D
            graphData={graphData}
            width={width}
            height={MAP_H}
            backgroundColor="#111827"
            nodeCanvasObject={paintNode}
            nodeCanvasObjectMode={() => 'replace'}
            nodeLabel={(n) =>
              `${n.id}${n.name ? ` — ${n.name}` : ''} (${n.type ?? n.labels?.[0] ?? ''})`
            }
            linkColor={() => 'rgba(75, 85, 99, 0.45)'}
            linkWidth={0.6}
            // Stop force simulation immediately — all nodes are geo-pinned via fx/fy
            cooldownTicks={0}
            d3AlphaDecay={1}
            d3VelocityDecay={1}
            enableNodeDrag={false}
            enableZoomInteraction
            onRenderFramePre={drawGrid}
          />
        </div>

        {/* Legend */}
        <div className="w-28 shrink-0 px-3 py-3 border-l border-gray-800 space-y-1.5 self-start mt-2">
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
