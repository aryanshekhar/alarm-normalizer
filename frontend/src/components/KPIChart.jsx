const SEV_COLOR = { high: '#ef4444', medium: '#f97316', low: '#eab308' };
const SEV_BADGE = {
  high:   'bg-red-950    text-red-300    border-red-800',
  medium: 'bg-orange-950 text-orange-300 border-orange-800',
  low:    'bg-yellow-950 text-yellow-300 border-yellow-800',
};

const TRAIN_STAGES = [
  { label: 'Preparing data',                   doneAt: 16  },
  { label: 'Learning normal traffic patterns',  doneAt: 35  },
  { label: 'Learning signal quality baselines', doneAt: 56  },
  { label: 'Learning capacity thresholds',      doneAt: 74  },
  { label: 'Calibrating anomaly sensitivity',   doneAt: 101 },
];

function stageState(stageDoneAt, pct) {
  const idx      = TRAIN_STAGES.findIndex((s) => s.doneAt === stageDoneAt);
  const prevDone = idx === 0 ? 0 : TRAIN_STAGES[idx - 1].doneAt;
  if (pct >= stageDoneAt) return 'done';
  if (pct >= prevDone)    return 'current';
  return 'pending';
}

// ── Training in-progress view ────────────────────────────────────────────────

function TrainingView({ progress }) {
  const pct = progress.value ?? progress.progress ?? 0;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Training AI Engine
        </h2>
        <span className="text-2xl font-bold text-blue-400">{pct}%</span>
      </div>

      <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden mb-4">
        <div
          className="h-full bg-blue-500 rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>

      <p className="text-sm font-medium text-gray-200 mb-4 min-h-[1.25rem]">
        {progress.message}
      </p>

      <ul className="space-y-1.5">
        {TRAIN_STAGES.map((stage) => {
          const state = stageState(stage.doneAt, pct);
          return (
            <li key={stage.label} className="flex items-center gap-2 text-xs">
              {state === 'done'    && <span className="text-green-400 w-4 shrink-0">✓</span>}
              {state === 'current' && <span className="text-yellow-400 w-4 shrink-0 animate-spin inline-block">⟳</span>}
              {state === 'pending' && <span className="text-gray-600 w-4 shrink-0">○</span>}
              <span className={
                state === 'done'    ? 'text-green-400' :
                state === 'current' ? 'text-yellow-300' : 'text-gray-600'
              }>
                {stage.label}
              </span>
            </li>
          );
        })}
      </ul>

      <div className="mt-4 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse inline-block" />
        <span className="text-xs text-gray-500">Training in progress&hellip;</span>
      </div>
    </div>
  );
}

// ── Training completion banner ────────────────────────────────────────────────

function CompletionView({ elapsed }) {
  return (
    <div className="bg-gray-900 rounded-lg border border-green-800 p-5">
      <div className="flex flex-col items-center justify-center gap-3 py-4">
        <span className="text-3xl">✅</span>
        <p className="text-base font-semibold text-green-300 text-center">
          AI engine armed — monitoring 47 cells across 4 domains
        </p>
        {elapsed > 0 && (
          <p className="text-xs text-gray-400">
            Training completed in {elapsed} second{elapsed === 1 ? '' : 's'}
          </p>
        )}
        <p className="text-xs text-gray-500 mt-1">Click Run Inference to analyse the network</p>
      </div>
    </div>
  );
}

// ── Correlation narrative view ────────────────────────────────────────────────

const STAGE_META = {
  checking:      { color: 'text-blue-400',   icon: '🔍' },
  no_alarms:     { color: 'text-green-400',  icon: '✅' },
  degrading:     { color: 'text-yellow-400', icon: '⚠️' },
  alarms_firing: { color: 'text-red-400',    icon: '🚨' },
  correlating:   { color: 'text-purple-400', icon: '⟲'  },
  complete:      { color: 'text-green-300',  icon: '✅' },
};

function CorrelationView({ correlationProgress, correlationResult }) {
  const prog = correlationProgress ?? correlationResult;
  if (!prog) return null;

  const { stage, message, progress = 0, alarm_count = 0 } = prog;
  const { color, icon } = STAGE_META[stage] ?? { color: 'text-gray-400', icon: '•' };
  const isComplete   = stage === 'complete';
  const alarmsFlash  = stage === 'alarms_firing' || stage === 'correlating' || isComplete;

  return (
    <div className={`bg-gray-900 rounded-lg border p-4 space-y-3 ${
      isComplete ? 'border-green-700' : alarmsFlash ? 'border-red-700' : 'border-gray-700'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Alarm Correlation
        </h2>
        {progress > 0 && (
          <span className="text-xs text-gray-500">{progress}%</span>
        )}
      </div>

      {/* Progress bar */}
      <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${
            isComplete ? 'bg-green-500' : alarmsFlash ? 'bg-red-500' : 'bg-blue-500'
          }`}
          style={{ width: `${Math.max(2, progress)}%` }}
        />
      </div>

      {/* Alarm storm banner */}
      {alarmsFlash && alarm_count > 0 && (
        <div className="flex items-center gap-2 px-3 py-2 rounded bg-red-950 border border-red-700 animate-pulse">
          <span className="text-red-400 font-bold text-sm">⚠️ {alarm_count} ALARMS FIRING</span>
        </div>
      )}

      {/* Current stage message */}
      <p className={`text-sm font-medium min-h-[1.25rem] ${color}`}>{message}</p>

      {/* Lead-time hero when complete */}
      {isComplete && prog.simba_lead_time_minutes != null && (
        <div className="mt-2 rounded bg-green-950 border border-green-700 px-4 py-3 text-center">
          <p className="text-2xl font-bold text-green-300">
            ✅ {prog.simba_lead_time_minutes} min early
          </p>
          <p className="text-xs text-green-500 mt-1">
            SIMBA detected this fault {prog.simba_lead_time_minutes} minutes before first alarm
          </p>
        </div>
      )}

      {/* Alarm cascade list when complete */}
      {isComplete && (prog.alarms ?? []).length > 0 && (
        <div className="space-y-1.5 max-h-52 overflow-y-auto pr-0.5 mt-1">
          {(prog.alarms ?? []).map((a) => (
            <div key={a.id} className="flex items-start gap-2 text-xs rounded bg-gray-800 border border-gray-700 px-2 py-1.5">
              <span className={`shrink-0 font-bold uppercase px-1 rounded text-[9px] ${
                a.severity === 'critical' ? 'bg-red-900 text-red-300' :
                a.severity === 'major'    ? 'bg-orange-900 text-orange-300' :
                                            'bg-yellow-900 text-yellow-300'
              }`}>{a.severity}</span>
              <div className="min-w-0">
                <span className="font-mono text-gray-200">{a.deviceId}</span>
                <span className="text-gray-500"> · {a.domain}</span>
                <p className="text-gray-500 truncate">{a.specificProblem.replace(/_/g, ' ')}</p>
              </div>
              {a.isRootCause && (
                <span className="ml-auto shrink-0 text-[9px] px-1 rounded bg-yellow-900 text-yellow-300 font-bold">ROOT</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Anomaly list ─────────────────────────────────────────────────────────────

function SeverityBadge({ severity }) {
  const cls = SEV_BADGE[severity] ?? 'bg-gray-800 text-gray-300 border-gray-700';
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded border text-[10px] font-bold uppercase ${cls}`}>
      {severity}
    </span>
  );
}

function AnomalyListView({ inferenceResult }) {
  const { anomalies, anomaly_count, timestamp } = inferenceResult;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Detected Anomalies
        </h2>
        <span className="text-xs text-gray-500">
          {anomaly_count} anomal{anomaly_count === 1 ? 'y' : 'ies'}
        </span>
      </div>

      <div className="space-y-2 max-h-72 overflow-y-auto pr-0.5">
        {anomalies.map((a) => {
          const topKpis = Object.entries(a.kpi_values ?? {}).slice(0, 4);
          const confPct = Math.round(a.confidence * 100);
          return (
            <div
              key={a.cell_id}
              className="rounded border border-gray-700 bg-gray-800 p-2.5"
            >
              {/* Row 1: cell/gNB + severity badge */}
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-mono font-semibold text-gray-100">
                  {a.cell_id}
                  <span className="text-gray-500 font-normal"> · {a.gnb_id}</span>
                </span>
                <SeverityBadge severity={a.severity} />
              </div>

              {/* Row 2: fault type + confidence */}
              <div className="flex items-center gap-2 mb-2">
                <span className="text-[11px] text-gray-400">
                  {a.fault_type.replace(/_/g, ' ')}
                </span>
                <span
                  className="text-[11px] font-semibold"
                  style={{ color: SEV_COLOR[a.severity] ?? '#6b7280' }}
                >
                  {confPct}% conf
                </span>
              </div>

              {/* Row 3: KPI values */}
              {topKpis.length > 0 && (
                <div className="flex flex-wrap gap-x-3 gap-y-0.5">
                  {topKpis.map(([name, val]) => (
                    <span key={name} className="text-[10px] text-gray-500">
                      {name.replace(/_/g, ' ')}
                      {' '}
                      <span className="text-gray-300">{Number(val).toFixed(1)}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <p className="text-[10px] text-gray-600 mt-2">{timestamp}</p>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function KPIChart({
  inferenceResult,
  trainingProgress,
  correlationProgress,
  correlationResult,
}) {
  // 1. Training in progress — always takes priority
  if (trainingProgress && trainingProgress.stage !== 'complete') {
    return <TrainingView progress={trainingProgress} />;
  }

  // 2. Correlation in progress (SSE stream active)
  if (correlationProgress) {
    return <CorrelationView correlationProgress={correlationProgress} correlationResult={null} />;
  }

  // 3. Correlation complete — show lead-time hero + alarm cascade
  if (correlationResult?.stage === 'complete') {
    return <CorrelationView correlationProgress={null} correlationResult={correlationResult} />;
  }

  // 4. Inference has run and found anomalies
  if (inferenceResult?.anomalies?.length > 0) {
    return <AnomalyListView inferenceResult={inferenceResult} />;
  }

  // 5. Inference has run but network is clean
  if (inferenceResult != null) {
    return (
      <div className="bg-gray-900 rounded-lg border border-green-900 p-4 h-36 flex flex-col items-center justify-center gap-2">
        <span className="text-2xl">✅</span>
        <p className="text-sm font-semibold text-green-400">
          Network healthy — no anomalies detected
        </p>
        <p className="text-[10px] text-gray-600">{inferenceResult.timestamp}</p>
      </div>
    );
  }

  // 6. Training just completed, waiting for inference
  if (trainingProgress?.stage === 'complete') {
    return <CompletionView elapsed={trainingProgress.elapsed} />;
  }

  // 7. Idle — nothing has run yet
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 h-36 flex items-center justify-center">
      <p className="text-gray-600 text-sm">Click Run Inference to analyse the network</p>
    </div>
  );
}
