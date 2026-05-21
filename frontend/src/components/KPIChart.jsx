import { useState } from 'react';

function fmtHHMMSS(isoStr) {
  if (!isoStr) return '--:--:--';
  try { return new Date(isoStr).toTimeString().slice(0, 8); } catch { return '--:--:--'; }
}

function addMinutes(isoStr, mins) {
  if (!isoStr || mins == null) return null;
  return new Date(new Date(isoStr).getTime() + mins * 60_000).toISOString();
}

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
                state === 'done'    ? 'text-slate-300' :
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
    <div className="bg-gray-900 rounded-lg border border-slate-700 p-5">
      <div className="flex flex-col items-center justify-center gap-3 py-4">
        <span className="text-3xl">✅</span>
        <p className="text-base font-semibold text-slate-100 text-center">
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
  no_alarms:     { color: 'text-slate-300',  icon: '✅' },
  degrading:     { color: 'text-yellow-400', icon: '⚠️' },
  alarms_firing: { color: 'text-red-400',    icon: '🚨' },
  correlating:   { color: 'text-purple-400', icon: '⟲'  },
  complete:      { color: 'text-slate-200',  icon: '✅' },
};

function CorrelationView({ correlationProgress, correlationResult, inferenceTimestamp }) {
  const [showAll, setShowAll] = useState(false);
  const prog = correlationProgress ?? correlationResult;
  if (!prog) return null;

  const { stage, message, progress = 0, alarm_count = 0 } = prog;
  const { color } = STAGE_META[stage] ?? { color: 'text-gray-400' };
  const isComplete  = stage === 'complete';
  const alarmsFlash = stage === 'alarms_firing' || stage === 'correlating' || isComplete;

  const allAlarms     = prog.alarms ?? [];
  const displayAlarms = showAll ? allAlarms : allAlarms.filter((a) => a.isRootCause);
  const leadMin       = prog.simba_lead_time_minutes;

  const t0Time    = fmtHHMMSS(inferenceTimestamp);
  const tAlarmTime = fmtHHMMSS(addMinutes(inferenceTimestamp, leadMin));

  return (
    <div className={`bg-gray-900 rounded-lg border p-4 space-y-3 ${
      isComplete ? 'border-slate-600' : alarmsFlash ? 'border-red-700' : 'border-gray-700'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Alarm Correlation
        </h2>
        {progress > 0 && <span className="text-xs text-gray-500">{progress}%</span>}
      </div>

      {/* Progress bar */}
      <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${
            isComplete ? 'bg-blue-600' : alarmsFlash ? 'bg-red-500' : 'bg-blue-500'
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

      {/* Stage message */}
      <p className={`text-sm font-medium min-h-[1.25rem] ${color}`}>{message}</p>

      {/* Lead-time hero + timeline */}
      {isComplete && leadMin != null && (
        <div className="rounded bg-slate-800 border border-slate-600 px-4 py-3 space-y-2">
          <div className="text-center">
            <p className="text-2xl font-bold text-slate-100">
              +{leadMin > 0 ? `${leadMin} min` : '< 1 min'} early
            </p>
            <p className="text-xs text-blue-300 mt-0.5 leading-relaxed">
              ML Model detected network degradation{' '}
              {leadMin >= 2 ? `${leadMin} minutes` : 'several minutes'}{' '}
              before the first alarm fired — providing sufficient lead time for proactive
              intervention before customer impact
            </p>
          </div>

          {/* Timeline graphic */}
          <div className="flex items-center gap-2 pt-1">
            <div className="flex flex-col items-center shrink-0 text-center">
              <span className="font-bold text-blue-300 text-[10px] leading-none">ML Model Alert</span>
              <span className="text-slate-400 text-[9px] font-mono">{t0Time}</span>
            </div>
            <div className="flex-1 flex flex-col items-center gap-0.5">
              <span className="text-blue-300 text-[10px] font-mono">
                + {leadMin >= 2 ? `${leadMin} min` : '< 2 min'}
              </span>
              <div className="w-full flex items-center gap-0.5">
                <div className="flex-1 border-t-2 border-dashed border-slate-500" />
                <span className="text-slate-400 text-xs">►</span>
              </div>
            </div>
            <div className="flex flex-col items-center shrink-0 text-center">
              <span className="font-bold text-red-400 text-[10px] leading-none">First Alarm</span>
              <span className="text-slate-400 text-[9px] font-mono">{tAlarmTime}</span>
            </div>
          </div>
        </div>
      )}

      {/* Suppression summary */}
      {isComplete && prog.total_alarms != null && (
        <div className="rounded bg-gray-800 border border-gray-700 px-3 py-2 text-center space-y-0.5">
          <div className="text-xs">
            <span className="text-orange-300 font-semibold">{prog.total_alarms} alarms</span>
            <span className="text-gray-500"> → </span>
            <span className="text-red-300 font-semibold">
              {prog.root_cause_alarm_count ?? 1} root cause
            </span>
            <span className="text-gray-500"> → </span>
            <span className="text-gray-400 font-semibold">
              {prog.suppressed_count ?? 0} suppressed
            </span>
          </div>
          <p className="text-[10px] text-gray-600">
            {prog.suppression_logic ?? 'Propagation-based deduplication'}
          </p>
        </div>
      )}

      {/* Alarm list with show-all toggle */}
      {isComplete && allAlarms.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-gray-500">
              {showAll ? `All ${allAlarms.length} alarms` : 'Root cause only'}
            </span>
            <button
              onClick={() => setShowAll((v) => !v)}
              className="text-[10px] px-2 py-0.5 rounded border border-gray-600 bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors"
            >
              {showAll ? 'Show root cause' : 'Show all'}
            </button>
          </div>

          <div className="max-h-52 overflow-y-auto pr-0.5 space-y-1.5">
            {displayAlarms.map((a) => (
              <div
                key={a.id}
                className="flex items-start gap-2 text-xs rounded bg-gray-800 border border-gray-700 px-2 py-1.5"
              >
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
                <div className="ml-auto shrink-0 flex gap-1">
                  {a.isRootCause && (
                    <span className="text-[9px] px-1 rounded bg-yellow-900 text-yellow-300 font-bold">ROOT</span>
                  )}
                  {!a.isRootCause && showAll && (
                    <span className="text-[9px] px-1 rounded bg-gray-700 text-gray-400 font-bold">SUP</span>
                  )}
                </div>
              </div>
            ))}
          </div>
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

// ── RCA view ──────────────────────────────────────────────────────────────────

const CONF_META = {
  low:    { pct: 33, bar: 'bg-yellow-500', text: 'text-yellow-400', label: 'Low' },
  medium: { pct: 66, bar: 'bg-orange-500', text: 'text-orange-400', label: 'Medium' },
  high:   { pct: 95, bar: 'bg-blue-600',   text: 'text-blue-300',   label: 'High' },
};

function RcaView({ rcaResult }) {
  const {
    rca_text,
    recommended_action,
    confidence = 'medium',
    affected_cells = [],
    propagation_path = [],
  } = rcaResult;

  const conf = CONF_META[confidence] ?? CONF_META.medium;

  // Build a deduplicated step chain from propagation_path edges
  const steps = [];
  if (propagation_path.length > 0) {
    steps.push(propagation_path[0].from_alarm);
    for (const edge of propagation_path) {
      if (steps[steps.length - 1] !== edge.to_alarm) steps.push(edge.to_alarm);
    }
  }

  // Map alarm ID → domain for labels
  const domainOf = {};
  for (const e of propagation_path) {
    domainOf[e.from_alarm] = e.from_domain;
    domainOf[e.to_alarm]   = e.to_domain;
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-purple-800 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Root Cause Analysis
        </h2>
        <span className={`text-xs font-semibold ${conf.text}`}>{conf.label} confidence</span>
      </div>

      {/* Confidence bar */}
      <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${conf.bar}`}
          style={{ width: `${conf.pct}%` }}
        />
      </div>

      {/* RCA text */}
      <p className="text-sm text-gray-200 leading-relaxed">{rca_text}</p>

      {/* Recommended action */}
      {recommended_action && (
        <div className="rounded bg-blue-950 border border-blue-800 px-3 py-2.5">
          <p className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider mb-1">
            Recommended Action
          </p>
          <p className="text-xs text-blue-200 leading-relaxed">{recommended_action}</p>
        </div>
      )}

      {/* Propagation chain */}
      {steps.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Propagation Path
          </p>
          <ol className="space-y-1">
            {steps.map((alarmId, i) => (
              <li key={alarmId} className="flex items-start gap-2 text-xs">
                <span className="shrink-0 w-4 text-right text-gray-600">{i + 1}.</span>
                <div>
                  <span className="font-mono text-gray-100">{alarmId}</span>
                  {domainOf[alarmId] && (
                    <span className="text-gray-500"> · {domainOf[alarmId]}</span>
                  )}
                  {i < steps.length - 1 && (
                    <div className="text-gray-700 text-[10px] ml-0 mt-0.5">↓</div>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Affected cells */}
      {affected_cells.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1.5">
            Affected Cells
          </p>
          <div className="flex flex-wrap gap-1">
            {affected_cells.map((c) => (
              <span
                key={c}
                className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700 text-gray-300"
              >
                {c}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function KPIChart({
  inferenceResult,
  trainingProgress,
  correlationProgress,
  correlationResult,
  rcaResult,
}) {
  const inferenceTimestamp = inferenceResult?.timestamp ?? null;

  // Determine the primary view (priority order)
  let primary;

  if (trainingProgress && trainingProgress.stage !== 'complete') {
    primary = <TrainingView progress={trainingProgress} />;
  } else if (correlationProgress) {
    primary = <CorrelationView correlationProgress={correlationProgress} correlationResult={null} inferenceTimestamp={inferenceTimestamp} />;
  } else if (correlationResult?.stage === 'complete') {
    primary = <CorrelationView correlationProgress={null} correlationResult={correlationResult} inferenceTimestamp={inferenceTimestamp} />;
  } else if (inferenceResult?.anomalies?.length > 0) {
    primary = <AnomalyListView inferenceResult={inferenceResult} />;
  } else if (inferenceResult != null) {
    primary = (
      <div className="bg-gray-900 rounded-lg border border-slate-700 p-4 h-36 flex flex-col items-center justify-center gap-2">
        <span className="text-2xl">✅</span>
        <p className="text-sm font-semibold text-slate-200">
          Network healthy — no anomalies detected
        </p>
        <p className="text-[10px] text-gray-600">{inferenceResult.timestamp}</p>
      </div>
    );
  } else if (trainingProgress?.stage === 'complete') {
    primary = <CompletionView elapsed={trainingProgress.elapsed} />;
  } else {
    primary = (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 h-36 flex items-center justify-center">
        <p className="text-gray-600 text-sm">Click Run Inference to analyse the network</p>
      </div>
    );
  }

  return (
    <>
      {primary}
      {rcaResult && <RcaView rcaResult={rcaResult} />}
    </>
  );
}
