import { useState } from 'react';
import { flushSync } from 'react-dom';
import {
  getTopology,
  trainModel,
  runInference,
  correlateAlarms,
  getRca,
} from '../api/mcpClient.js';

const STAGES = [
  { id: 'topology',  label: 'Load Topology',   desc: 'Fetch network graph from Neo4j' },
  { id: 'train',     label: 'Train Model',      desc: 'Train SIMBA on 30-day KPI window' },
  { id: 'inference', label: 'Run Inference',    desc: 'Detect anomalies in current window' },
  { id: 'correlate', label: 'Correlate Alarms', desc: 'Group related alarms by propagation' },
  { id: 'rca',       label: 'Get RCA',          desc: 'Root cause analysis via LLM' },
  { id: 'monitor',   label: 'Start Monitor',    desc: 'Connect WebSocket for live events' },
];

export default function DemoControls({
  onTopology,
  onInference,
  onCorrelation,
  onRca,
  onConnectWs,
  onTrainingProgress,
  wsConnected,
  inferenceResult,
  correlationResult,
}) {
  const [stageStatus, setStageStatus] = useState({});
  const [error, setError] = useState(null);

  function setStatus(id, status) {
    setStageStatus((s) => ({ ...s, [id]: status }));
  }

  async function runStage(id) {
    setStatus(id, 'loading');
    setError(null);
    try {
      switch (id) {
        case 'topology': {
          const data = await getTopology();
          onTopology(data);
          break;
        }
        case 'train': {
          flushSync(() =>
            onTrainingProgress({ value: 0, stage: 'preparing', message: 'Starting training...' }),
          );
          for await (const event of trainModel(5, 30, true)) {
            if (event.stage === 'error') throw new Error(event.message);
            if (event.progress >= 0) {
              const normalized = { value: event.progress, stage: event.stage, message: event.message };
              console.log('[onTrainingProgress]', normalized);
              flushSync(() => onTrainingProgress(normalized));
            }
          }
          onTrainingProgress(null);
          break;
        }
        case 'inference': {
          const data = await runInference('anomalous');
          onInference(data);
          break;
        }
        case 'correlate': {
          const cellIds = inferenceResult?.anomalies?.map((a) => a.cell_id) ?? [];
          const data = await correlateAlarms(cellIds);
          onCorrelation(data);
          break;
        }
        case 'rca': {
          const anomalyIds = inferenceResult?.anomalies?.map((a) => a.cell_id) ?? [];
          const alarmIds = correlationResult?.groups
            ?.flatMap((g) => g.alarms?.map((a) => a.id).filter(Boolean) ?? []) ?? [];
          const data = await getRca('', anomalyIds, alarmIds);
          onRca(data);
          break;
        }
        case 'monitor': {
          onConnectWs();
          break;
        }
      }
      setStatus(id, 'done');
    } catch (e) {
      setStatus(id, 'error');
      setError(e.message);
      if (id === 'train') onTrainingProgress(null);
    }
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 space-y-3">
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Demo Stages
      </h2>

      {STAGES.map((stage) => {
        const status = stageStatus[stage.id] ?? 'idle';
        const isMonitor = stage.id === 'monitor';
        const isDone = isMonitor ? wsConnected : status === 'done';
        const isLoading = status === 'loading';
        const isError = status === 'error';

        let cls =
          'w-full text-left px-3 py-2 rounded border text-sm transition-colors disabled:cursor-not-allowed ';
        if (isDone)         cls += 'border-green-700 bg-green-950 text-green-300';
        else if (isError)   cls += 'border-red-700 bg-red-950 text-red-300';
        else if (isLoading) cls += 'border-yellow-700 bg-yellow-950 text-yellow-300 animate-pulse';
        else                cls += 'border-gray-700 bg-gray-800 text-gray-200 hover:border-gray-500';

        return (
          <button
            key={stage.id}
            onClick={() => runStage(stage.id)}
            disabled={isLoading || (isMonitor && wsConnected)}
            className={cls}
          >
            <div className="font-medium">{stage.label}</div>
            <div className="text-xs opacity-60 mt-0.5">{stage.desc}</div>
          </button>
        );
      })}

      {error && (
        <p className="text-xs text-red-400 bg-red-950 border border-red-800 rounded p-2 break-words">
          {error}
        </p>
      )}
    </div>
  );
}
