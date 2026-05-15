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

export default function KPIChart({ inferenceResult }) {
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
