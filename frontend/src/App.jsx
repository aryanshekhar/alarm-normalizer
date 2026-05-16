import { useState, useRef, useCallback, useMemo } from 'react';
import DemoControls from './components/DemoControls.jsx';
import KPIChart from './components/KPIChart.jsx';
import TopologyMap from './components/TopologyMap.jsx';
import NOCAssistant from './components/NOCAssistant.jsx';
import AgentLog from './components/AgentLog.jsx';
import { createMonitorSocket } from './api/mcpClient.js';

// Mini icon strip shown when the left panel is collapsed
const STAGE_ICONS = ['⬡', '⚙', '⚡', '⟲', '🔍', '📡'];

export default function App() {
  const [topology, setTopology]             = useState(null);
  const [inferenceResult, setInference]     = useState(null);
  const [correlationResult, setCorrelation] = useState(null);
  const [rcaResult, setRca]                 = useState(null);
  const [wsEvents, setWsEvents]             = useState([]);
  const [wsConnected, setWsConnected]       = useState(false);
  const [leftCollapsed, setLeftCollapsed]       = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [correlationProgress, setCorrelationProgress] = useState(null);
  const trainingStartRef = useRef(null);
  const wsRef = useRef(null);

  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    wsRef.current = createMonitorSocket(
      (event) => setWsEvents((prev) => [event, ...prev].slice(0, 100)),
      () => setWsConnected(true),
      () => setWsConnected(false),
    );
  }, []);

  const handleCorrelationProgress = useCallback((prog) => {
    setCorrelationProgress(prog);
  }, []);

  // All devices involved in the alarm storm (highlighted during / after correlation)
  const alarmingNodeIds = useMemo(() => {
    const stage = correlationProgress?.stage;
    if (stage === 'alarms_firing' || stage === 'correlating' || stage === 'complete') {
      return new Set(correlationProgress?.alarming_device_ids ?? []);
    }
    if (correlationResult?.alarming_device_ids) {
      return new Set(correlationResult.alarming_device_ids);
    }
    return new Set();
  }, [correlationProgress, correlationResult]);

  // Root-cause devices (known only after correlation is complete)
  const rootCauseNodeIds = useMemo(() => {
    if (correlationResult?.stage === 'complete') {
      return new Set(correlationResult.root_cause_device_ids ?? []);
    }
    return new Set();
  }, [correlationResult]);

  // Suppressed = alarming but not root cause (shown only after complete)
  const suppressedNodeIds = useMemo(() => {
    if (correlationResult?.stage === 'complete' && rootCauseNodeIds.size > 0) {
      const suppressed = new Set();
      for (const id of alarmingNodeIds) {
        if (!rootCauseNodeIds.has(id)) suppressed.add(id);
      }
      return suppressed;
    }
    return new Set();
  }, [correlationResult, alarmingNodeIds, rootCauseNodeIds]);

  const handleTrainingProgress = useCallback((prog) => {
    if (prog === null) {
      // Stream ended — record elapsed and show completion
      const elapsed = trainingStartRef.current
        ? Math.round((Date.now() - trainingStartRef.current) / 1000)
        : 0;
      trainingStartRef.current = null;
      setTrainingProgress({ stage: 'complete', elapsed });
    } else {
      if (trainingStartRef.current === null) {
        trainingStartRef.current = Date.now();
      }
      setTrainingProgress(prog);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-3 shrink-0">
        <span className="text-green-400 font-bold tracking-wide">AIOps</span>
        <span className="text-gray-400 text-sm">Agentic AI Canvas for AIOps</span>
        {wsConnected && (
          <span className="ml-auto text-xs text-green-400 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse inline-block" />
            Monitor live
          </span>
        )}
      </header>

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <div className="flex-1 flex gap-4 p-4 min-h-0 overflow-hidden">

        {/* Left panel — collapsible ──────────────────────────────────── */}
        <div
          className="shrink-0 relative transition-all duration-200"
          style={{ width: leftCollapsed ? '2rem' : '16rem' }}
        >
          {/* Collapse / expand toggle button on the right edge */}
          <button
            onClick={() => setLeftCollapsed((c) => !c)}
            title={leftCollapsed ? 'Expand controls' : 'Collapse controls'}
            className="absolute -right-3 top-3 z-20 w-6 h-6 bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-full text-xs text-gray-300 flex items-center justify-center shadow"
          >
            {leftCollapsed ? '›' : '‹'}
          </button>

          {/* Full controls */}
          {!leftCollapsed && (
            <DemoControls
              onTopology={setTopology}
              onInference={setInference}
              onCorrelation={setCorrelation}
              onCorrelationProgress={handleCorrelationProgress}
              onRca={setRca}
              onConnectWs={connectWs}
              onTrainingProgress={handleTrainingProgress}
              wsConnected={wsConnected}
              inferenceResult={inferenceResult}
              correlationResult={correlationResult}
            />
          )}

          {/* Collapsed icon strip */}
          {leftCollapsed && (
            <div className="flex flex-col items-center gap-3 pt-10">
              {STAGE_ICONS.map((icon, i) => (
                <span key={i} className="text-sm text-gray-600" title={`Stage ${i + 1}`}>
                  {icon}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Centre — topology + KPI ────────────────────────────────────── */}
        <div className="flex-1 min-w-0 space-y-4 overflow-y-auto">
          <TopologyMap
            topology={topology}
            alarmingNodeIds={alarmingNodeIds}
            rootCauseNodeIds={rootCauseNodeIds}
            suppressedNodeIds={suppressedNodeIds}
          />
          <KPIChart
            inferenceResult={inferenceResult}
            trainingProgress={trainingProgress}
            correlationProgress={correlationProgress}
            correlationResult={correlationResult}
          />
        </div>

        {/* Right panel — agent events + chat ─────────────────────────── */}
        <div className="shrink-0 w-64 space-y-4">
          <AgentLog events={wsEvents} />
          <NOCAssistant rcaResult={rcaResult} />
        </div>
      </div>
    </div>
  );
}
