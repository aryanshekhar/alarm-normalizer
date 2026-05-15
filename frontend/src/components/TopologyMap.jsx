import { useRef, useState, useLayoutEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const NODE_COLOR = {
  Cell:         '#22d3ee',
  RANNode:      '#3b82f6',
  OpticalNode:  '#8b5cf6',
  IPNode:       '#06b6d4',
  Host:         '#10b981',
  VNF:          '#6ee7b7',
  Service:      '#f59e0b',
  NetworkSlice: '#fbbf24',
  Alarm:        '#ef4444',
};

function nodeColor(node) {
  return NODE_COLOR[node.labels?.[0]] ?? '#6b7280';
}

export default function TopologyMap({ topology }) {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(500);

  useLayoutEffect(() => {
    if (containerRef.current) {
      setWidth(containerRef.current.offsetWidth);
    }
  }, [topology]);

  if (!topology?.nodes?.length) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 h-72 flex items-center justify-center">
        <p className="text-gray-600 text-sm">Load topology to see network graph</p>
      </div>
    );
  }

  const graphData = {
    nodes: topology.nodes.map((n) => ({ ...n, id: n.id })),
    links: topology.edges.map((e) => ({ source: e.from, target: e.to, type: e.type })),
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      <div className="px-4 py-2 border-b border-gray-800 flex items-center justify-between">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Network Topology
        </h2>
        <span className="text-xs text-gray-500">
          {topology.node_count} nodes &middot; {topology.edge_count} edges
        </span>
      </div>
      <div ref={containerRef} style={{ height: 250 }}>
        <ForceGraph2D
          graphData={graphData}
          width={width}
          height={250}
          backgroundColor="#111827"
          nodeColor={nodeColor}
          nodeLabel={(n) => `${n.id} (${n.labels?.[0] ?? ''})`}
          linkColor={() => '#374151'}
          nodeRelSize={4}
          linkWidth={1}
          enableNodeDrag
        />
      </div>
    </div>
  );
}
