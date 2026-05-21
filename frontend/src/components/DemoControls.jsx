import { useState, useEffect } from 'react';
import { flushSync } from 'react-dom';
import {
  getTopology,
  trainModel,
  runInference,
  correlateAlarmsStream,
  getRca,
  getTrainingMode,
} from '../api/mcpClient.js';

const STAGES = [
  { id: 'topology',  label: 'Load Topology',   desc: 'Fetch network graph from Neo4j' },
  { id: 'train',     label: 'Train Model',      desc: 'Train ML Model on 30-day KPI window' },
  { id: 'inference', label: 'Run Inference',    desc: 'Detect anomalies in current window' },
  { id: 'correlate', label: 'Monitor & Correlate Alarms', desc: 'Group related alarms by propagation' },
  { id: 'rca',       label: 'Get RCA',          desc: 'Root cause analysis via LLM' },
];

export default function DemoControls({
  onTopology,
  onInference,
  onCorrelation,
  onCorrelationProgress,
  onRca,
  onTrainingProgress,
  inferenceResult,
  correlationResult,
}) {
  const [stageStatus, setStageStatus] = useState({});
  const [error, setError] = useState(null);
  const [trainingMode, setTrainingMode] = useState(null);

  useEffect(() => {
    getTrainingMode().then((r) => setTrainingMode(r.mode)).catch(() => {});
  }, []);

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
          console.log('[DemoControls] calling onInference with', data?.anomaly_count, 'anomalies');
          onInference(data);
          break;
        }
        case 'correlate': {
          const cellIds = inferenceResult?.anomalies?.map((a) => a.cell_id) ?? [];
          flushSync(() =>
            onCorrelationProgress({ stage: 'checking', progress: 5, message: 'Initialising correlation...', alarms: [] }),
          );
          for await (const event of correlateAlarmsStream(cellIds)) {
            if (event.stage === 'error') throw new Error(event.message);
            flushSync(() => onCorrelationProgress(event));
            if (event.stage === 'complete') {
              flushSync(() => onCorrelation(event));
            }
          }
          onCorrelationProgress(null);
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
      }
      setStatus(id, 'done');
    } catch (e) {
      setStatus(id, 'error');
      setError(e.message);
      if (id === 'train') onTrainingProgress(null);
      if (id === 'correlate') onCorrelationProgress(null);
    }
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 space-y-3">
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
        Demo Stages
      </h2>

      {STAGES.map((stage) => {
        const status = stageStatus[stage.id] ?? 'idle';
        const isDone = status === 'done';
        const isLoading = status === 'loading';
        const isError = status === 'error';

        let cls =
          'w-full text-left px-3 py-2 rounded border text-sm transition-colors disabled:cursor-not-allowed ';
        if (isDone)         cls += 'border-slate-600 bg-slate-700 text-slate-200';
        else if (isError)   cls += 'border-red-700 bg-red-950 text-red-300';
        else if (isLoading) cls += 'border-yellow-700 bg-yellow-950 text-yellow-300 animate-pulse';
        else                cls += 'border-gray-700 bg-gray-800 text-gray-200 hover:border-gray-500';

        return (
          <button
            key={stage.id}
            onClick={() => runStage(stage.id)}
            disabled={isLoading}
            className={cls}
          >
            <div className="font-medium flex items-center gap-2">
              {stage.label}
              {stage.id === 'train' && trainingMode && (
                <span className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded border ${
                  trainingMode === 'demo'
                    ? 'border-gray-600 bg-gray-800 text-gray-400'
                    : 'border-blue-700 bg-blue-950 text-blue-300'
                }`}>
                  {trainingMode} mode
                </span>
              )}
            </div>
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
