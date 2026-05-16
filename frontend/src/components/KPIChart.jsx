import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const SEV_COLOR = { high: '#ef4444', medium: '#f97316', low: '#eab308' };

const TRAIN_STAGES = [
  { label: 'Preparing data',                   doneAt: 40  },
  { label: 'Learning traffic patterns',         doneAt: 65  },
  { label: 'Learning signal quality baselines', doneAt: 85  },
  { label: 'Learning capacity thresholds',      doneAt: 100 },
  { label: 'Arming AI engine',                  doneAt: 101 },
];

function stageState(stageDoneAt, pct) {
  const idx      = TRAIN_STAGES.findIndex((s) => s.doneAt === stageDoneAt);
  const prevDone = idx === 0 ? 0 : TRAIN_STAGES[idx - 1].doneAt;
  if (pct >= stageDoneAt) return 'done';
  if (pct >= prevDone)    return 'current';
  return 'pending';
}

function TrainingView({ progress }) {
  // Backend SSE events use "progress"; normalized objects use "value".
  const pct = progress.value ?? progress.progress ?? 0;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-5 animate-pulse-slow">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Training AI Engine
        </h2>
        <span className="text-2xl font-bold text-blue-400">{pct}%</span>
      </div>

      {/* Large progress bar */}
      <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden mb-4">
        <div
          className="h-full bg-blue-500 rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>

      {/* Stage message */}
      <p className="text-sm font-medium text-gray-200 mb-4 min-h-[1.25rem]">
        {progress.message}
      </p>

      {/* Stage checklist */}
      <ul className="space-y-1.5">
        {TRAIN_STAGES.map((stage) => {
          const state = stageState(stage.doneAt, pct);
          return (
            <li key={stage.label} className="flex items-center gap-2 text-xs">
              {state === 'done' && (
                <span className="text-green-400 w-4 shrink-0">✓</span>
              )}
              {state === 'current' && (
                <span className="text-yellow-400 w-4 shrink-0 animate-spin inline-block">⟳</span>
              )}
              {state === 'pending' && (
                <span className="text-gray-600 w-4 shrink-0">○</span>
              )}
              <span
                className={
                  state === 'done'
                    ? 'text-green-400'
                    : state === 'current'
                    ? 'text-yellow-300'
                    : 'text-gray-600'
                }
              >
                {stage.label}
              </span>
            </li>
          );
        })}
      </ul>

      {/* Alive pulse indicator */}
      <div className="mt-4 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse inline-block" />
        <span className="text-xs text-gray-500">Training in progress&hellip;</span>
      </div>
    </div>
  );
}

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
      </div>
    </div>
  );
}

export default function KPIChart({ inferenceResult, trainingProgress }) {
  // Training in progress — always show progress view
  if (trainingProgress && trainingProgress.stage !== 'complete') {
    return <TrainingView progress={trainingProgress} />;
  }

  // Training just completed, no inference result yet — show completion banner
  if (trainingProgress?.stage === 'complete' && !inferenceResult?.anomalies?.length) {
    return <CompletionView elapsed={trainingProgress.elapsed} />;
  }

  // Normal KPI view
  if (!inferenceResult?.anomalies?.length) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 h-52 flex items-center justify-center">
        <p className="text-gray-600 text-sm">Run inference to see KPI snapshot</p>
      </div>
    );
  }

  // Show KPI values for the highest-confidence anomaly
  const anomaly = [...inferenceResult.anomalies].sort(
    (a, b) => b.confidence - a.confidence,
  )[0];

  const data = Object.entries(anomaly.kpi_values).map(([name, value]) => ({
    name: name.replace(/_/g, ' '),
    value: parseFloat(Number(value).toFixed(3)),
  }));

  const color = SEV_COLOR[anomaly.severity] ?? '#6b7280';

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          KPI Snapshot
        </h2>
        <span className="text-xs text-gray-500">
          {anomaly.cell_id} &middot; {anomaly.fault_type} &middot; conf{' '}
          {anomaly.confidence}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={190}>
        <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 28 }}>
          <XAxis
            dataKey="name"
            tick={{ fill: '#6b7280', fontSize: 9 }}
            angle={-35}
            textAnchor="end"
            interval={0}
          />
          <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
          <Tooltip
            contentStyle={{
              background: '#111827',
              border: '1px solid #374151',
              fontSize: 11,
            }}
            labelStyle={{ color: '#d1d5db' }}
          />
          <Bar dataKey="value" radius={[3, 3, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="text-xs text-gray-600 mt-1">
        {inferenceResult.anomaly_count} anomal
        {inferenceResult.anomaly_count === 1 ? 'y' : 'ies'} detected &middot;{' '}
        {inferenceResult.timestamp}
      </p>
    </div>
  );
}
