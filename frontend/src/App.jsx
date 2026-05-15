import { useState, useRef, useCallback } from 'react';
import DemoControls from './components/DemoControls.jsx';
import KPIChart from './components/KPIChart.jsx';
import TopologyMap from './components/TopologyMap.jsx';
import NOCAssistant from './components/NOCAssistant.jsx';
import AgentLog from './components/AgentLog.jsx';
import { createMonitorSocket } from './api/mcpClient.js';

export default function App() {
  const [topology, setTopology] = useState(null);
  const [inferenceResult, setInferenceResult] = useState(null);
  const [correlationResult, setCorrelationResult] = useState(null);
  const [rcaResult, setRcaResult] = useState(null);
  const [wsEvents, setWsEvents] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);

  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    wsRef.current = createMonitorSocket(
      (event) => setWsEvents((prev) => [event, ...prev].slice(0, 100)),
      () => setWsConnected(true),
      () => setWsConnected(false),
    );
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-3 shrink-0">
        <span className="text-green-400 font-bold tracking-wide">AIOps</span>
        <span className="text-gray-600 text-sm">Alarm Normalizer</span>
        {wsConnected && (
          <span className="ml-auto text-xs text-green-400 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse inline-block" />
            Monitor live
          </span>
        )}
      </header>

      <div className="flex-1 grid grid-cols-12 gap-4 p-4 min-h-0">
        {/* Left: demo stage controls */}
        <div className="col-span-3 space-y-4">
          <DemoControls
            onTopology={setTopology}
            onInference={setInferenceResult}
            onCorrelation={setCorrelationResult}
            onRca={setRcaResult}
            onConnectWs={connectWs}
            wsConnected={wsConnected}
            inferenceResult={inferenceResult}
            correlationResult={correlationResult}
          />
        </div>

        {/* Centre: topology graph + KPI chart */}
        <div className="col-span-6 space-y-4">
          <TopologyMap topology={topology} />
          <KPIChart inferenceResult={inferenceResult} />
        </div>

        {/* Right: live agent events + NOC chat */}
        <div className="col-span-3 space-y-4">
          <AgentLog events={wsEvents} />
          <NOCAssistant rcaResult={rcaResult} />
        </div>
      </div>
    </div>
  );
}
